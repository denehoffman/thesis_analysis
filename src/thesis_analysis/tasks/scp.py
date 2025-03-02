import luigi
from paramiko import SSHClient
from scp import SCPClient
from thesis_analysis.constants import global_parameters
from thesis_analysis.logger import logger


class SCP(luigi.Task):
    remote_path = luigi.Parameter()
    local_path = luigi.Parameter()

    resources = {'scp': 1}

    def run(self):
        with SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.connect(
                str(global_parameters().hostname),
                username=str(global_parameters().username),
            )

            transport = ssh.get_transport()
            assert transport is not None
            with SCPClient(transport) as scp:
                logger.info(f'Copying {self.remote_path} to {self.local_path}')
                scp.get(
                    remote_path=str(self.remote_path),
                    local_path=str(self.local_path),
                )

    def output(self):
        return [luigi.LocalTarget(self.local_path)]
