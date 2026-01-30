import sys
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# -------------------------------
# READ FROM POSTGRES (TEST)
# -------------------------------
dyf = glueContext.create_dynamic_frame.from_options(
    connection_type="postgresql",
    connection_options={
        "url": "jdbc:postgresql://titan-database-1.chkeu0qa25jq.us-east-1.rds.amazonaws.com:5432/titanibdai2025",
        "dbtable": "public.store_snapshot",
        "secretId": "titan-rds-db-credentials-meera"
    }
)

print("Row count:", dyf.count())

# -------------------------------
# WRITE TO S3 AS CSV
# -------------------------------
glueContext.write_dynamic_frame.from_options(
    frame=dyf,
    connection_type="s3",
    connection_options={
        "path": "s3://titan-glue-test-data/postgres_check/",
        "partitionKeys": []
    },
    format="csv"
)

job.commit()
