import errno
from itertools import chain
import locale
import logging
import os
import shutil
import signal
import socket
import subprocess
import tempfile
from types import TracebackType
from typing import (
    Any,
    BinaryIO,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)
from ..utils.macaddr import randommac
from .console import ConsoleSocket

logger = logging.getLogger(__name__)


class QEMUMachineError(Exception):
    """
    Exception called when an error in QEMUMachine happens.
    """


class QEMUMachineAddDeviceError(QEMUMachineError):
    """
    Exception raised when a request to add a device can not be fulfilled
    The failures are caused by limitations, lack of information or conflicting
    requests on the QEMUMachine methods.  This exception does not represent
    failures reported by the QEMU binary itself.
    """


class VMLaunchFailure(QEMUMachineError):
    """
    Exception raised when a VM launch was attempted, but failed.
    """
    def __init__(self, exitcode: Optional[int],
                 command: str, output: Optional[str]):
        super().__init__(exitcode, command, output)
        self.exitcode = exitcode
        self.command = command
        self.output = output

    def __str__(self) -> str:
        ret = ''
        if self.__cause__ is not None:
            name = type(self.__cause__).__name__
            reason = str(self.__cause__)
            if reason:
                ret += f"{name}: {reason}"
            else:
                ret += f"{name}"
        ret += '\n'
        if self.exitcode is not None:
            ret += f"\tExit code: {self.exitcode}\n"
        ret += f"\tCommand: {self.command}\n"
        ret += f"\tOutput: {self.output}\n"
        return ret


class AbnormalShutdown(QEMUMachineError):
    """
    Exception raised when a graceful shutdown was requested, but not performed.
    """


_T = TypeVar('_T', bound='QEMUMachine')


class QEMUMachine:
    """
    A QEMU VM.
    Use this object as a context manager to ensure
    the QEMU process terminates::
        with VM(binary) as vm:
            ...
        # vm is guaranteed to be shut down here
    """
    def __init__(self,
                 binary: str,
                 args: Sequence[str] = (),
                 wrapper: Sequence[str] = (),
                 name: Optional[str] = None,
                 base_temp_dir: str = "/var/tmp",
                 sock_dir: Optional[str] = None,
                 console_log: Optional[str] = None,
                 log_dir: Optional[str] = None):
        """
        Initialize a QEMUMachine
        """
        # Direct user configuration
        self._binary = binary
        self._args = list(args)
        self._wrapper = wrapper
        self._name = name or f"qemu-{os.getpid()}-{id(self):02x}"
        self._temp_dir: Optional[str] = None
        self._base_temp_dir = base_temp_dir
        self._sock_dir = sock_dir
        self._log_dir = log_dir
        self._console_log_path = console_log

        # Runstate
        self._qemu_log_path: Optional[str] = None
        self._qemu_log_file: Optional[BinaryIO] = None
        self._popen: Optional['subprocess.Popen[bytes]'] = None
        self._iolog: Optional[str] = None
        self._qemu_full_args: Tuple[str, ...] = ()
        self._launched = False
        self._machine: Optional[str] = None
        self._cpu: Optional[str] = None
        self._disks: List[dict] = []
        self._nets: List[dict] = []
        self._portforward = False
        self._host_port = None
        self._guest_port = None
        self._kernel_args: List[str] = []
        self._console_index = 0
        self._console_set = False
        self._console_device_type: Optional[str] = None
        self._console_address = os.path.join(
            self.sock_dir, f"{self._name}-console.sock"
        )
        self._console_socket: Optional[socket.socket] = None
        self._remove_files: List[str] = []
        self._user_killed = False
        self._quit_issued = False

    def __enter__(self: _T) -> _T:
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        self.shutdown()

    def add_fd(self: _T, fd: int, fdset: int,
               opaque: str, opts: str = '') -> _T:
        """
        Pass a file descriptor to the VM
        """
        options = ['fd=%d' % fd,
                   'set=%d' % fdset,
                   'opaque=%s' % opaque]
        if opts:
            options.append(opts)
        # This did not exist before 3.4, but since then it is
        # mandatory for our purpose
        if hasattr(os, 'set_inheritable'):
            os.set_inheritable(fd, True)
        self._args.append('-add-fd')
        self._args.append(','.join(options))
        return self

    @staticmethod
    def _remove_if_exists(path: str) -> None:
        """
        Remove file object at path if it exists
        """
        try:
            os.remove(path)
        except OSError as exception:
            if exception.errno == errno.ENOENT:
                return
            raise

    def is_running(self) -> bool:
        """Returns true if the VM is running."""
        return self._popen is not None and self._popen.poll() is None

    @property
    def _subp(self) -> 'subprocess.Popen[bytes]':
        if self._popen is None:
            raise QEMUMachineError('Subprocess pipe not present')
        return self._popen

    def exitcode(self) -> Optional[int]:
        """Returns the exit code if possible, or None."""
        if self._popen is None:
            return None
        return self._popen.poll()

    def get_pid(self) -> Optional[int]:
        """Returns the PID of the running process, or None."""
        if not self.is_running():
            return None
        return self._subp.pid

    def _load_io_log(self) -> None:
        # Assume that the output encoding of QEMU's terminal output is
        # defined by our locale. If indeterminate, allow open() to fall
        # back to the platform default.
        _, encoding = locale.getlocale()
        if self._qemu_log_path is not None:
            with open(self._qemu_log_path, "r", encoding=encoding) as iolog:
                self._iolog = iolog.read()

    @property
    def _base_args(self) -> List[str]:
        # defaults for micro virtual machines
        args = ['-nodefaults', '-no-user-config', '-no-reboot', '-no-acpi', '-nographic', '-device',
                'virtio-serial-device']

        if self._machine is not None:
            args.extend(['-machine', self._machine])

        if self._disks is not None:
            for index in range(len(self._disks)):
                args.extend(['-drive',
                             f'file={self._disks[index]["path"]},'
                             f'if=none,id=drive{index},'
                             f'format={self._disks[index]["type"]}'])
                args.extend(['-device', f'virtio-blk-device,drive=drive{index}'])

        if self._nets is not None:
            for index in range(len(self._nets)):
                if self._nets[index]["type"] == "tap":
                    args.extend(['-netdev',
                                 f'tap,id=net{index},ifname={self._nets[index]["ifname"]},'
                                 'script=no,downscript=no,'
                                 '-device', f'virtio-net-device,netdev=net{index},mac={self._nets[index]["mac"]}'])
                elif self._nets[index]["type"] == "user":
                    if self._portforward:
                        args.extend(['-netdev',
                                     f'user,id=net{index},hostfwd=tcp::{self._host_port}-:{self._guest_port}',
                                     '-device', f'virtio-net-device,netdev=net{index},mac={self._nets[index]["mac"]}'])
                    else:
                        args.extend(['-netdev',
                                     f'user,id=net{index}',
                                     '-device', f'virtio-net-device,netdev=net{index},mac={self._nets[index]["mac"]}'])

        if self._kernel_args is not None:
            args.extend(self._kernel_args)

        for _ in range(self._console_index):
            args.extend(['-serial', 'null'])

        if self._console_set:
            chardev = ('socket,id=virticon0,path=%s,server=on,wait=off' %
                       self._console_address)
            args.extend(['-chardev', chardev])
            if self._console_device_type is None:
                args.extend(['-serial', 'chardev:console'])
            else:
                device = '%s,chardev=virticon0' % self._console_device_type
                args.extend(['-device', device])

        if self._cpu is not None:
            args.extend(['-cpu', self._cpu])
        else:
            args.extend(['-cpu', 'host'])

        return args

    @property
    def args(self) -> List[str]:
        """Returns the list of arguments given to the QEMU binary."""
        return self._args

    def _pre_launch(self) -> None:
        if self._console_set:
            self._remove_files.append(self._console_address)

        # NOTE: Make sure any opened resources are *definitely* freed in
        # _post_shutdown()!
        self._qemu_log_path = os.path.join(self.log_dir, self._name + ".log")
        self._qemu_log_file = open(self._qemu_log_path, 'wb')
        self._iolog = None
        self._qemu_full_args = tuple(chain(
            self._wrapper,
            [self._binary],
            self._base_args,
            self._args
        ))

    def _close_qemu_log_file(self) -> None:
        if self._qemu_log_file is not None:
            self._qemu_log_file.close()
            self._qemu_log_file = None

    def _post_shutdown(self) -> None:
        """
        Called to cleanup the VM instance after the process has exited.
        May also be called after a failed launch.
        """
        logger.debug("Cleaning up after VM process")
        self._close_qemu_log_file()
        self._load_io_log()
        self._qemu_log_path = None
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
        while len(self._remove_files) > 0:
            self._remove_if_exists(self._remove_files.pop())
        exitcode = self.exitcode()
        if (exitcode is not None and exitcode < 0
                and not (self._user_killed and exitcode == -signal.SIGKILL)):
            msg = 'qemu received signal %i; command: "%s"'
            if self._qemu_full_args:
                command = ' '.join(self._qemu_full_args)
            else:
                command = ''
            logger.warning(msg, -int(exitcode), command)
        self._quit_issued = False
        self._user_killed = False
        self._launched = False

    def launch(self) -> None:
        """
        Launch the VM and make sure we cleanup and expose the
        command line/output in case of exception
        """
        if self._launched:
            raise QEMUMachineError('VM already launched')
        try:
            self._launch()
        except BaseException as exc:
            # We may have launched the process but it may
            # have exited before we could connect via QMP.
            # Assume the VM didn't launch or is exiting.
            # If we don't wait for the process, exitcode() may still be
            # 'None' by the time control is ceded back to the caller.
            if self._launched:
                self.wait()
            else:
                self._post_shutdown()
            if isinstance(exc, Exception):
                raise VMLaunchFailure(
                    exitcode=self.exitcode(),
                    command=' '.join(self._qemu_full_args),
                    output=self._iolog
                ) from exc
            # Don't wrap 'BaseException'; doing so would downgrade
            # that exception. However, we still want to clean up.
            raise

    def _launch(self) -> None:
        """
        Launch the VM
        """
        self._pre_launch()
        logger.debug('VM launch command: %r', ' '.join(self._qemu_full_args))
        # Cleaning up of this subprocess is guaranteed by _do_shutdown.
        self._popen = subprocess.Popen(self._qemu_full_args,
                                       stdin=subprocess.DEVNULL,
                                       stdout=self._qemu_log_file,
                                       stderr=subprocess.STDOUT,
                                       shell=False,
                                       close_fds=False)
        self._launched = True

    def _early_cleanup(self) -> None:
        """
        Cleanup the VM before we try to kill it
        """
        if self._console_socket is not None:
            logger.debug('Closing console socket')
            self._console_socket.close()
            self._console_socket = None

    def _hard_shutdown(self) -> None:
        """
        Perform early cleanup, kill the VM, and wait for it to terminate.
        """
        logger.debug("Performing hard shutdown")
        self._early_cleanup()
        self._subp.kill()
        self._subp.wait(timeout=60)

    def _soft_shutdown(self, timeout: Optional[int]) -> None:
        """
        Perform early cleanup, attempt to gracefully shut down the VM, and wait
        for it to terminate.
        """
        logger.debug("Attempting graceful termination")
        self._early_cleanup()

        if self._quit_issued:
            logger.debug(
                "Anticipating QEMU termination due to prior 'quit' command, "
                "or explicit call to wait()"
            )
        else:
            logger.debug("Politely asking QEMU to terminate")
        if not self._quit_issued:
            logger.debug(
                "Not anticipating QEMU quit and no QMP connection present, "
                "issuing SIGTERM"
            )
            self._subp.terminate()
        # May raise subprocess.TimeoutExpired
        logger.debug(
            "Waiting (timeout=%s) for QEMU process (pid=%s) to terminate",
            timeout, self._subp.pid
        )
        self._subp.wait(timeout=timeout)

    def _do_shutdown(self, timeout: Optional[int]) -> None:
        """
        Attempt to shutdown the VM gracefully; fallback to a hard shutdown.
        """
        try:
            self._soft_shutdown(timeout)
        except Exception as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                logger.debug("Timed out waiting for QEMU process to exit")
            logger.debug("Graceful shutdown failed", exc_info=True)
            logger.debug("Falling back to hard shutdown")
            self._hard_shutdown()
            raise AbnormalShutdown("Could not perform graceful shutdown") \
                from exc

    def shutdown(self,
                 hard: bool = False,
                 timeout: Optional[int] = 30) -> None:
        """
        Terminate the VM (gracefully if possible) and perform cleanup.
        """
        if not self._launched:
            return
        logger.debug("Shutting down VM appliance; timeout=%s", timeout)
        if hard:
            logger.debug("Caller requests immediate termination of QEMU process.")
        try:
            if hard:
                self._user_killed = True
                self._hard_shutdown()
            else:
                self._do_shutdown(timeout)
        finally:
            self._post_shutdown()

    def kill(self) -> None:
        """
        Terminate the VM forcefully, wait for it to exit, and perform cleanup.
        """
        self.shutdown(hard=True)

    def wait(self, timeout: Optional[int] = 30) -> None:
        """
        Wait for the VM to power off and perform post-shutdown cleanup.
        """
        self._quit_issued = True
        self.shutdown(timeout=timeout)

    @staticmethod
    def event_match(event: Any, match: Optional[Any]) -> bool:
        """
        Check if an event matches optional match criteria.
        """
        if match is None:
            return True
        try:
            for key in match:
                if key in event:
                    if not QEMUMachine.event_match(event[key], match[key]):
                        return False
                else:
                    return False
            return True
        except TypeError:
            # either match or event wasn't iterable (not a dict)
            return bool(match == event)

    def get_log(self) -> Optional[str]:
        """
        After self.shutdown or failed qemu execution, this returns the output
        of the qemu process.
        """
        return self._iolog

    def add_args(self, *args: str) -> None:
        """
        Adds to the list of extra arguments to be given to the QEMU binary
        """
        self._args.extend(args)

    def set_machine(self, machine_type: str) -> None:
        """
        Sets the machine type
        If set, the machine type will be added to the base arguments
        of the resulting QEMU command line.
        """
        self._machine = machine_type

    def set_cpu(self, cpu_type: str) -> None:
        """
        Sets the CPU type
        If set, the CPU type will be added to the base arguments
        of the resulting QEMU command line.
        """
        self._cpu = cpu_type

    def add_virtio_disk(self, disk_path: str, disk_type: str) -> None:
        """
        Adds a virtio disk to the list of disks to be attached to the VM
        """
        if not os.path.exists(disk_path):
            raise QEMUMachineError(f'File {disk_path} does not exist')
        elif not os.path.isfile(disk_path):
            raise QEMUMachineError(f'{disk_path} is not a file')

        if disk_type not in ('raw', 'qcow2'):
            raise QEMUMachineError(f'Unsupported disk type {disk_type}')

        new_disk = {'path': disk_path, 'type': disk_type}
        self._disks.append(new_disk)

    def add_virtio_net(self, net_type: str, net_if: str = None) -> None:
        """
        Adds a virtio network interface to the VM
        """
        if net_type not in ('user', 'tap'):
            raise QEMUMachineError(f'Unsupported net type {net_type}')

        if net_type == 'tap' and net_if is None:
            raise QEMUMachineError('TAP network interface must be specified')

        new_net = {'type': net_type, 'if': net_if, 'mac': randommac()}
        self._nets.append(new_net)

    def portforward(self, host_port: int, guest_port: int) -> None:
        """
        Sets up port forwarding from the host to the guest
        """
        self._portforward = True
        self._host_port = host_port
        self._guest_port = guest_port

    def set_console(self,
                    device_type: Optional[str] = None,
                    console_index: int = 0) -> None:
        """
        Sets the device type for a console device
        """
        self._console_set = True
        self._console_device_type = device_type
        self._console_index = console_index

    def add_kernel(self, kernel_path: str, kernel_append: Optional[str] = None) -> None:
        """
        Adds a kernel to the list of kernels to be attached to the VM
        """
        if not os.path.exists(kernel_path):
            raise QEMUMachineError(f'File {kernel_path} does not exist')
        elif not os.path.isfile(kernel_path):
            raise QEMUMachineError(f'{kernel_path} is not a file')

        self._kernel_args.extend(['-kernel', kernel_path])
        if kernel_append is not None:
            self._kernel_args.extend(['-append', kernel_append])

    @property
    def console_socket(self) -> socket.socket:
        """
        Returns the socket used to communicate with the console device
        """
        if self._console_socket is None:
            self._console_socket = ConsoleSocket(
                self._console_address,
                file=self._console_log_path,
                drain=True,
            )
        return self._console_socket

    @property
    def temp_dir(self) -> str:
        """
        Returns a temporary directory to be used for this machine
        """
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="qemu-machine-",
                                              dir=self._base_temp_dir)
        return self._temp_dir

    @property
    def sock_dir(self) -> str:
        """
        Returns the directory used for sockfiles by this machine.
        """
        if self._sock_dir:
            return self._sock_dir
        return self.temp_dir

    @property
    def log_dir(self) -> str:
        """
        Returns a directory to be used for writing logs
        """
        if self._log_dir is None:
            return self.temp_dir
        return self._log_dir
