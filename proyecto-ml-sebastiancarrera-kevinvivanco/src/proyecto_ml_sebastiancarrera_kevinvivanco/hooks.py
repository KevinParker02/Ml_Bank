from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession
from dotenv import load_dotenv 

class EnvironmentVariableHooks:
    """Carga variables de entorno desde .env automáticamente."""

    @hook_impl
    def after_context_created(self, context) -> None:
        """Carga las variables del archivo .env antes de inicializar Spark u otros recursos."""
        load_dotenv()  # 👈 carga las variables del .env
        print("✅ Variables de entorno cargadas desde .env")


class SparkHooks:
    @hook_impl
    def after_context_created(self, context) -> None:
        """Initialises a SparkSession using the config
        defined in project's conf folder.
        """

        # Load the spark configuration in spark.yaml using the config loader
        parameters = context.config_loader["spark"]
        spark_conf = SparkConf().setAll(parameters.items())

        # Initialise the spark session
        spark_session_conf = (
            SparkSession.builder.appName(context.project_path.name)
            .enableHiveSupport()
            .config(conf=spark_conf)
        )
        _spark_session = spark_session_conf.getOrCreate()
        _spark_session.sparkContext.setLogLevel("WARN")
