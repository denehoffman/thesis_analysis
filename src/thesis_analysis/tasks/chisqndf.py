import luigi


class ChiSqNDF(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()
    chisqndf = luigi.FloatParameter()

    def requires(self):
        return []
