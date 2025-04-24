from typing import final, override

import luigi
from paramiko import SSHClient
from scp import SCPClient

from thesis_analysis.constants import GLUEX_HOSTNAME, GLUEX_USERNAME
from thesis_analysis.logger import logger


@final
class SCP(luigi.Task):
    remote_path = luigi.Parameter()
    local_path = luigi.Parameter()

    resources = {'scp': 1}

    @override
    def run(self):
        with SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.connect(
                GLUEX_HOSTNAME,
                username=GLUEX_USERNAME,
            )

            transport = ssh.get_transport()
            assert transport is not None
            with SCPClient(transport) as scp:
                logger.info(f'Copying {self.remote_path} to {self.local_path}')
                scp.get(
                    remote_path=str(self.remote_path),
                    local_path=str(self.local_path),
                )

    @override
    def output(self):
        return [luigi.LocalTarget(self.local_path)]
