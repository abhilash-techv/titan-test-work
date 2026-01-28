import sys
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

from pyspark.sql.functions import (
    col, lower, to_date,
    min as spark_min,
    max as spark_max,
    count,
    countDistinct,
    sum as spark_sum,
    sequence, explode,
    lit, row_number,
    year, month, dayofmonth,
    datediff, when, trunc,
    concat, regexp_replace, trim
)
from pyspark.sql.window import Window

# --------------------------------------------------
# Glue setup
# --------------------------------------------------
args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# --------------------------------------------------
# Read tables
# --------------------------------------------------
member_df = glueContext.create_dynamic_frame.from_catalog(
    database="titan-final-db",
    table_name="member_data"
).toDF()

txn_df = glueContext.create_dynamic_frame.from_catalog(
    database="titan-final-db",
    table_name="transaction_data"
).toDF()

# --------------------------------------------------
# DATE SPINE
# --------------------------------------------------
member_dates = (
    member_df
    .filter(
        lower(col("enrollment_channel_code"))
        .isin("tanishq", "encircle", "encirclewebsite", "ecommtanishq")
    )
    .withColumn("enrollment_date", to_date(col("enrollment_date")))
    .select("enrollment_date")
    .filter(col("enrollment_date").isNotNull())
)

txn_dates = (
    txn_df
    .filter(lower(col("channel")) == "tanishq")
    .withColumn("invoice_date", to_date(col("invoice_date")))
    .select("invoice_date")
    .filter(col("invoice_date").isNotNull())
)

max_member_date = member_dates.agg(spark_max("enrollment_date")).collect()[0][0]
max_txn_date = txn_dates.agg(spark_max("invoice_date")).collect()[0][0]

max_date = max(max_member_date, max_txn_date)
min_date = lit("2021-04-01").cast("date")

date_df = (
    spark.range(1)
    .select(explode(sequence(min_date, lit(max_date))).alias("date"))
)

# --------------------------------------------------
# BASE TRANSACTIONS (NORMALIZED CARD_NO)
# --------------------------------------------------
txn_base = (
    txn_df
    .filter(lower(col("channel")) == "tanishq")
    .withColumn(
        "card_no",
        trim(regexp_replace(col("card_no").cast("string"), "\\.0$", ""))
    )
    .withColumn("invoice_date", to_date(col("invoice_date")))
    .filter(col("invoice_date").isNotNull())
    .filter(col("card_no").isNotNull())
)

# --------------------------------------------------
# MEMBER BASE (NORMALIZED CARD_NO)
# --------------------------------------------------
member_base = (
    member_df
    .filter(
        lower(col("enrollment_channel_code"))
        .isin("tanishq", "encircle", "encirclewebsite", "ecommtanishq")
    )
    .withColumn(
        "card_no",
        trim(regexp_replace(col("card_no").cast("string"), "\\.0$", ""))
    )
)

# --------------------------------------------------
# FIRST PURCHASE
# --------------------------------------------------
first_purchase_df = (
    txn_base
    .groupBy("card_no")
    .agg(spark_min("invoice_date").alias("first_purchase_date"))
)

# --------------------------------------------------
# NEW CUSTOMER COUNT
# --------------------------------------------------
new_customer_df = (
    first_purchase_df
    .groupBy("first_purchase_date")
    .agg(count("card_no").alias("newcustomercount"))
    .withColumnRenamed("first_purchase_date", "date")
)

# --------------------------------------------------
# REPEAT CUSTOMER COUNT
# --------------------------------------------------
repeat_txns = (
    txn_base
    .join(first_purchase_df, on="card_no", how="left")
    .filter(col("invoice_date") > col("first_purchase_date"))
)

repeat_customer_df = (
    repeat_txns
    .groupBy("invoice_date")
    .agg(countDistinct("card_no").alias("repeatcustomercount"))
    .withColumnRenamed("invoice_date", "date")
)

# --------------------------------------------------
# NEW CUSTOMER SALES
# --------------------------------------------------
new_customer_sales_df = (
    txn_base
    .join(first_purchase_df, on="card_no", how="left")
    .filter(col("invoice_date") == col("first_purchase_date"))
    .filter(col("eligible_amt").isNotNull())
    .groupBy("invoice_date")
    .agg(
        spark_sum(col("eligible_amt").cast("double"))
        .alias("newcustomersales")
    )
    .withColumnRenamed("invoice_date", "date")
)

# --------------------------------------------------
# REPEAT CUSTOMER SALES
# --------------------------------------------------
repeat_customer_sales_df = (
    repeat_txns
    .filter(col("eligible_amt").isNotNull())
    .groupBy("invoice_date")
    .agg(
        spark_sum(col("eligible_amt").cast("double"))
        .alias("repeatcustomersales")
    )
    .withColumnRenamed("invoice_date", "date")
)

# --------------------------------------------------
# BIRTHDAY CUSTOMER BASE (±15 days)
# --------------------------------------------------
birthday_base = (
    txn_base
    .join(
        member_base.withColumn("dob", to_date(col("dob"))),
        on="card_no",
        how="inner"
    )
    .filter(col("dob").isNotNull())
    .withColumn(
        "event_this_year",
        when(
            (month(col("dob")) == 2) & (dayofmonth(col("dob")) == 29),
            to_date(concat(lit("28-02-"), year(col("invoice_date"))), "dd-MM-yyyy")
        ).otherwise(
            to_date(
                concat(
                    dayofmonth(col("dob")),
                    lit("-"),
                    month(col("dob")),
                    lit("-"),
                    year(col("invoice_date"))
                ),
                "d-M-yyyy"
            )
        )
    )
    .withColumn("days_diff", datediff(col("invoice_date"), col("event_this_year")))
    .filter((col("days_diff") >= -15) & (col("days_diff") <= 15))
)

birthday_customer_df = (
    birthday_base
    .withColumn("date", trunc(col("invoice_date"), "DAY"))
    .groupBy("date")
    .agg(countDistinct("card_no").alias("birthdaycustomercount"))
)

birthday_customer_sales_df = (
    birthday_base
    .filter(col("eligible_amt").isNotNull())
    .groupBy(trunc(col("invoice_date"), "DAY").alias("date"))
    .agg(
        spark_sum(col("eligible_amt").cast("double"))
        .alias("birthdaycustomersales")
    )
)

# --------------------------------------------------
# ANNIVERSARY CUSTOMER BASE (±15 days)
# --------------------------------------------------
anniversary_base = (
    txn_base
    .join(
        member_base.withColumn("anniversary", to_date(col("anniversary"))),
        on="card_no",
        how="inner"
    )
    .filter(col("anniversary").isNotNull())
    .withColumn(
        "event_this_year",
        to_date(
            concat(
                year(col("invoice_date")),
                lit("-"),
                month(col("anniversary")),
                lit("-"),
                when(dayofmonth(col("anniversary")) > 28, 28)
                .otherwise(dayofmonth(col("anniversary")))
            ),
            "yyyy-M-d"
        )
    )
    .withColumn("days_diff", datediff(col("invoice_date"), col("event_this_year")))
    .filter((col("days_diff") >= -15) & (col("days_diff") <= 15))
)

anniversary_customer_df = (
    anniversary_base
    .withColumn("date", trunc(col("invoice_date"), "DAY"))
    .groupBy("date")
    .agg(countDistinct("card_no").alias("anncustomercount"))
)

anniversary_customer_sales_df = (
    anniversary_base
    .filter(col("eligible_amt").isNotNull())
    .groupBy(trunc(col("invoice_date"), "DAY").alias("date"))
    .agg(
        spark_sum(col("eligible_amt").cast("double"))
        .alias("annvcustomersales")
    )
)

# --------------------------------------------------
# CONSOLIDATED DATA
# --------------------------------------------------
consolidated_df = (
    date_df
    .join(new_customer_df, on="date", how="left")
    .join(repeat_customer_df, on="date", how="left")
    .join(new_customer_sales_df, on="date", how="left")
    .join(repeat_customer_sales_df, on="date", how="left")
    .join(birthday_customer_df, on="date", how="left")
    .join(anniversary_customer_df, on="date", how="left")
    .join(birthday_customer_sales_df, on="date", how="left")
    .join(anniversary_customer_sales_df, on="date", how="left")
    .fillna({
        "newcustomercount": 0,
        "repeatcustomercount": 0,
        "newcustomersales": 0.0,
        "repeatcustomersales": 0.0,
        "birthdaycustomercount": 0,
        "anncustomercount": 0,
        "birthdaycustomersales": 0.0,
        "annvcustomersales": 0.0
    })
)

# --------------------------------------------------
# ADD ID
# --------------------------------------------------
window_spec = Window.orderBy(col("date").desc())

consolidated_df = (
    consolidated_df
    .withColumn("id", row_number().over(window_spec))
    .select(
        "id",
        "date",
        "newcustomercount",
        "repeatcustomercount",
        "newcustomersales",
        "repeatcustomersales",
        "birthdaycustomercount",
        "anncustomercount",
        "birthdaycustomersales",
        "annvcustomersales"
    )
)

# --------------------------------------------------
# WRITE OUTPUT
# --------------------------------------------------
(
    consolidated_df
    .write
    .mode("overwrite")
    .format("parquet")
    .save("s3://titan-glue-test-data/consolidated-test/consolidated_data/")
)

job.commit()
