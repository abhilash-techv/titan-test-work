import sys
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

from pyspark.sql.functions import (
    col, lower, to_date,
    min as spark_min, max as spark_max,
    count, countDistinct, sum as spark_sum,
    sequence, explode, lit, row_number,
    year, month, dayofmonth,
    datediff, when, trunc,
    concat, regexp_replace, trim,
    add_months
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
    .filter(lower(col("enrollment_channel_code"))
            .isin("tanishq", "encircle", "encirclewebsite", "ecommtanishq"))
    .withColumn("enrollment_date", to_date(col("enrollment_date")))
    .filter(col("enrollment_date").isNotNull())
    .select("enrollment_date")
)

txn_dates = (
    txn_df
    .filter(lower(col("channel")) == "tanishq")
    .withColumn("invoice_date", to_date(col("invoice_date")))
    .filter(col("invoice_date").isNotNull())
    .select("invoice_date")
)

max_date = max(
    member_dates.agg(spark_max("enrollment_date")).collect()[0][0],
    txn_dates.agg(spark_max("invoice_date")).collect()[0][0]
)

min_date = lit("2021-04-01").cast("date")

date_df = spark.range(1).select(
    explode(sequence(min_date, lit(max_date))).alias("date")
)

# --------------------------------------------------
# BASE TRANSACTIONS (NORMALIZED CARD_NO)
# --------------------------------------------------
txn_base = (
    txn_df
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
    .filter(lower(col("enrollment_channel_code"))
            .isin("tanishq", "encircle", "encirclewebsite", "ecommtanishq"))
    .withColumn(
        "card_no",
        trim(regexp_replace(col("card_no").cast("string"), "\\.0$", ""))
    )
    .withColumn("enrollment_date", to_date(col("enrollment_date")))
    .filter(col("enrollment_date").isNotNull())
)

# --------------------------------------------------
# FIRST TANISHQ TRANSACTION PER CUSTOMER
# --------------------------------------------------
first_tanishq_txn = (
    txn_base
    .filter(lower(col("channel")) == "tanishq")
    .groupBy("card_no")
    .agg(spark_min("invoice_date").alias("first_ta_date"))
)

# --------------------------------------------------
# PRIOR OTHER-CHANNEL TRANSACTIONS
# --------------------------------------------------
prior_other_txn = (
    txn_base
    .filter(lower(col("channel")) != "tanishq")
    .join(first_tanishq_txn, "card_no", "inner")
    .filter(col("invoice_date") < col("first_ta_date"))
    .select("card_no")
    .distinct()
)

# --------------------------------------------------
# ENROLLED BEFORE FIRST TANISHQ TRANSACTION
# --------------------------------------------------
enrolled_before_ta = (
    member_base
    .join(first_tanishq_txn, "card_no", "inner")
    .filter(col("enrollment_date") < col("first_ta_date"))
    .select("card_no")
    .distinct()
)

# --------------------------------------------------
# CROSS CHANNEL BUYERS
# --------------------------------------------------
cross_channel_buyers = (
    first_tanishq_txn
    .join(
        prior_other_txn.union(enrolled_before_ta),
        "card_no",
        "inner"
    )
)

cross_channel_buyers_df = (
    cross_channel_buyers
    .groupBy("first_ta_date")
    .agg(count("card_no").alias("crosschannelbuyers"))
    .withColumnRenamed("first_ta_date", "date")
)


# --------------------------------------------------
# LAST TRANSACTION PER CUSTOMER
# --------------------------------------------------
last_txn_df = (
    txn_base
    .groupBy("card_no")
    .agg(spark_max("invoice_date").alias("last_transaction_date"))
)

# --------------------------------------------------
# MEMBERS ARCHIVED PER DAY  ✅ NEW
# --------------------------------------------------
members_with_last_txn = (
    member_base
    .join(last_txn_df, "card_no", "left")
)

members_archived_df = (
    date_df
    .join(
        members_with_last_txn,
        members_with_last_txn.enrollment_date <= col("date"),
        "left"
    )
    .withColumn(
        "threshold_date",
        add_months(col("date"), -36)
    )
    .filter(
        (col("last_transaction_date").isNull()) |
        (col("last_transaction_date") < col("threshold_date"))
    )
    .groupBy("date")
    .agg(countDistinct("card_no").alias("membersarchived"))
)


# --------------------------------------------------
# DAILY ENROLLED CUSTOMERS  ✅ NEW
# --------------------------------------------------
daily_enrolled_df = (
    member_base
    .withColumn("enrollment_date", to_date(col("enrollment_date")))
    .filter(col("enrollment_date").isNotNull())
    .groupBy("enrollment_date")
    .agg(countDistinct("card_no").alias("enrolledcustomercount"))
    .withColumnRenamed("enrollment_date", "date")
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
# CORE METRICS
# --------------------------------------------------
new_customer_df = (
    first_purchase_df
    .groupBy("first_purchase_date")
    .agg(count("card_no").alias("newcustomercount"))
    .withColumnRenamed("first_purchase_date", "date")
)

repeat_txns = (
    txn_base
    .join(first_purchase_df, "card_no", "left")
    .filter(col("invoice_date") > col("first_purchase_date"))
)

repeat_customer_df = (
    repeat_txns
    .groupBy("invoice_date")
    .agg(countDistinct("card_no").alias("repeatcustomercount"))
    .withColumnRenamed("invoice_date", "date")
)

total_customer_txn_df = (
    txn_base
    .groupBy("invoice_date")
    .agg(countDistinct("card_no").alias("totalcustomertransactions"))
    .withColumnRenamed("invoice_date", "date")
)

new_customer_sales_df = (
    txn_base
    .join(first_purchase_df, "card_no", "left")
    .filter(col("invoice_date") == col("first_purchase_date"))
    .filter(col("eligible_amt").isNotNull())
    .groupBy("invoice_date")
    .agg(spark_sum(col("eligible_amt").cast("double"))
         .alias("newcustomersales"))
    .withColumnRenamed("invoice_date", "date")
)

repeat_customer_sales_df = (
    repeat_txns
    .filter(col("eligible_amt").isNotNull())
    .groupBy("invoice_date")
    .agg(spark_sum(col("eligible_amt").cast("double"))
         .alias("repeatcustomersales"))
    .withColumnRenamed("invoice_date", "date")
)

transacted_customer_sales_df = (
    txn_base
    .filter(col("eligible_amt").isNotNull())
    .groupBy("invoice_date")
    .agg(
        spark_sum(col("eligible_amt").cast("double"))
        .alias("transactedcustomersales")
    )
    .withColumnRenamed("invoice_date", "date")
)

# --------------------------------------------------
# BIRTHDAY METRICS (transaction-based ±15 days)
# --------------------------------------------------
birthday_base = (
    txn_base
    .join(member_base.withColumn("dob", to_date(col("dob"))),
          "card_no", "inner")
    .filter(col("dob").isNotNull())
    .withColumn(
        "event_this_year",
        when(
            (month(col("dob")) == 2) & (dayofmonth(col("dob")) == 29),
            to_date(concat(lit("28-02-"), year(col("invoice_date"))), "dd-MM-yyyy")
        ).otherwise(
            to_date(concat(
                dayofmonth(col("dob")), lit("-"),
                month(col("dob")), lit("-"),
                year(col("invoice_date"))
            ), "d-M-yyyy")
        )
    )
    .withColumn(
        "days_diff",
        datediff(col("invoice_date"), col("event_this_year"))
    )
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
# ANNIVERSARY METRICS (transaction-based ±15 days)
# --------------------------------------------------
anniversary_base = (
    txn_base
    .join(member_base.withColumn("anniversary", to_date(col("anniversary"))),
          "card_no", "inner")
    .filter(col("anniversary").isNotNull())
    .withColumn(
        "event_this_year",
        to_date(concat(
            year(col("invoice_date")), lit("-"),
            month(col("anniversary")), lit("-"),
            when(dayofmonth(col("anniversary")) > 28, 28)
            .otherwise(dayofmonth(col("anniversary")))
        ), "yyyy-M-d")
    )
    .withColumn(
        "days_diff",
        datediff(col("invoice_date"), col("event_this_year"))
    )
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
# CUSTOMER BIRTHDAYS PER DAY (enrollment-based)
# --------------------------------------------------
member_birthdays = (
    member_base
    .withColumn("dob", to_date(col("dob")))
    .withColumn("enrollment_date", to_date(col("enrollment_date")))
    .filter(col("dob").isNotNull())
    .filter(col("enrollment_date").isNotNull())
    .withColumn("birth_month", month(col("dob")))
    .withColumn("birth_day", dayofmonth(col("dob")))
    .withColumn(
        "enroll_ym",
        concat(year(col("enrollment_date")), lit("-"), month(col("enrollment_date")))
    )
)

date_with_ym = (
    date_df
    .withColumn("year_m", year(col("date")))
    .withColumn("month_m", month(col("date")))
    .withColumn("day_m", dayofmonth(col("date")))
    .withColumn(
        "date_ym",
        concat(year(col("date")), lit("-"), month(col("date")))
    )
)

customer_birthdays_perday_df = (
    date_with_ym
    .join(
        member_birthdays,
        (member_birthdays.enroll_ym <= col("date_ym")) &
        (member_birthdays.birth_month == col("month_m")) &
        (member_birthdays.birth_day == col("day_m")),
        "left"
    )
    .groupBy("date")
    .agg(countDistinct("card_no").alias("customerbirthdaysperday"))
)

# --------------------------------------------------
# CUSTOMER ANNIVERSARIES PER DAY (enrollment-based)  ✅ NEW
# --------------------------------------------------
member_anniversaries = (
    member_base
    .withColumn("anniversary", to_date(col("anniversary")))
    .withColumn("enrollment_date", to_date(col("enrollment_date")))
    .filter(col("anniversary").isNotNull())
    .filter(col("enrollment_date").isNotNull())
    .withColumn("anniv_month", month(col("anniversary")))
    .withColumn("anniv_day", dayofmonth(col("anniversary")))
    .withColumn(
        "enroll_ym",
        concat(year(col("enrollment_date")), lit("-"), month(col("enrollment_date")))
    )
)

customer_anniversaries_perday_df = (
    date_with_ym
    .join(
        member_anniversaries,
        (member_anniversaries.enroll_ym <= col("date_ym")) &
        (member_anniversaries.anniv_month == col("month_m")) &
        (member_anniversaries.anniv_day == col("day_m")),
        "left"
    )
    .groupBy("date")
    .agg(countDistinct("card_no").alias("customeranniversariesperday"))
)

# --------------------------------------------------
# DIAMOND ENTHUSIASTS  ✅ NEW
# --------------------------------------------------
diamond_enthusiasts_df = (
    txn_base
    .filter(lower(col("channel")) == "tanishq")
    .filter(lower(col("category")).contains("diamond"))
    .join(
        member_base.select("card_no"),
        "card_no",
        "inner"
    )
    .groupBy("invoice_date")
    .agg(
        countDistinct("card_no").alias("diamondenthusiasts")
    )
    .withColumnRenamed("invoice_date", "date")
)

# --------------------------------------------------
# DORMANT CUSTOMERS  ✅ NEW
# --------------------------------------------------

# Last eligible Tanishq transaction per customer
last_txn_df = (
    txn_base
    .filter(col("eligible_amt") != 0)
    .filter(lower(col("channel")) == "tanishq")
    .groupBy("card_no")
    .agg(spark_max("invoice_date").alias("last_txn_date"))
)

# Eligible members (points > 50)
eligible_members_df = (
    member_base
    .filter(col("point_balance") > 50)
    .join(last_txn_df, "card_no", "left")
    .filter(col("last_txn_date").isNotNull())
)

# Dormant customers per day (last txn < date - 6 months)
dormant_customers_df = (
    date_df
    .join(
        eligible_members_df,
        eligible_members_df.last_txn_date < add_months(col("date"), -6),
        "left"
    )
    .groupBy("date")
    .agg(
        countDistinct("card_no").alias("dormantcustomers")
    )
)

# --------------------------------------------------
# LOW POINT BALANCE CUSTOMERS  ✅ NEW
# --------------------------------------------------

low_point_members_df = (
    member_base
    .withColumn("enrollment_date", to_date(col("enrollment_date")))
    .filter(col("point_balance") < 50)
)

low_point_balance_df = (
    date_df
    .join(
        low_point_members_df,
        low_point_members_df.enrollment_date <= col("date"),
        "left"
    )
    .groupBy("date")
    .agg(
        countDistinct("card_no").alias("lowpointbalancecustomers")
    )
)

# --------------------------------------------------
# ACTIVE CUSTOMERS  ✅ NEW
# --------------------------------------------------

# Active transaction base
active_txn_base = (
    txn_base
    .filter(col("eligible_amt") != 0)
    .filter(
        lower(col("channel")).isin(
            "tanishq", "encircle", "encirclewebsite", "ecommtanishq"
        )
    )
    .join(
        member_base.select("card_no", "enrollment_date"),
        "card_no",
        "inner"
    )
    .filter(col("invoice_date") >= col("enrollment_date"))
)

active_customers_df = (
    active_txn_base
    .groupBy("invoice_date")
    .agg(
        countDistinct("card_no").alias("activecustomers")
    )
    .withColumnRenamed("invoice_date", "date")
)

# --------------------------------------------------
# CAMPAIGNS DEPLOYED  ✅ FIXED
# --------------------------------------------------

campaign_df = glueContext.create_dynamic_frame.from_catalog(
    database="titan-final-db",
    table_name="campaign_data"
).toDF()

campaign_normalized_df = (
    campaign_df
    .withColumn(
        "deploy_date",
        when(
            col("deployment_date").rlike("^[0-9]{4}-[0-9]{2}-[0-9]{2}"),
            to_date(col("deployment_date"))
        ).when(
            col("deployment_date").rlike("^[0-9]{2}-[0-9]{2}-[0-9]{4}"),
            to_date(col("deployment_date"), "dd-MM-yyyy")
        ).otherwise(None)
    )
    .filter(col("deploy_date").isNotNull())
)

campaigns_deployed_df = (
    campaign_normalized_df
    .groupBy("deploy_date")
    .agg(
        count("*").alias("campaignsdeployed")
    )
    .withColumnRenamed("deploy_date", "date")
)

# --------------------------------------------------
# TARGETED COUNT  ✅ FIXED
# --------------------------------------------------

campaign_df = glueContext.create_dynamic_frame.from_catalog(
    database="titan-final-db",
    table_name="campaign_data"
).toDF()

campaign_target_df = (
    campaign_df
    .withColumn(
        "deploy_date",
        when(
            col("deployment_date").rlike("^[0-9]{2}-[0-9]{2}-[0-9]{4}"),
            to_date(col("deployment_date"), "dd-MM-yyyy")
        ).when(
            col("deployment_date").rlike("^[0-9]{4}-[0-9]{2}-[0-9]{2}"),
            to_date(col("deployment_date").substr(1, 10), "yyyy-MM-dd")
        ).otherwise(None)
    )
    .withColumn(
        "target_count_num",
        when(
            trim(col("target_count")).rlike("^[0-9]+$"),
            trim(col("target_count")).cast("int")
        ).otherwise(lit(0))
    )
    .filter(col("deploy_date").isNotNull())
)

targeted_count_df = (
    campaign_target_df
    .groupBy("deploy_date")
    .agg(
        spark_sum("target_count_num").alias("targetedcount")
    )
    .withColumnRenamed("deploy_date", "date")
)

# --------------------------------------------------
# CAMPAIGN BUYERS  ✅ NEW
# --------------------------------------------------

campaign_df = glueContext.create_dynamic_frame.from_catalog(
    database="titan-final-db",
    table_name="campaign_data"
).toDF()

campaign_buyers_df = (
    campaign_df
    .withColumn(
        "deploy_date",
        when(
            col("deployment_date").rlike("^[0-9]{2}-[0-9]{2}-[0-9]{4}"),
            to_date(col("deployment_date"), "dd-MM-yyyy")
        ).when(
            col("deployment_date").rlike("^[0-9]{4}-[0-9]{2}-[0-9]{2}"),
            to_date(col("deployment_date").substr(1, 10), "yyyy-MM-dd")
        ).otherwise(None)
    )
    .withColumn(
        "buyers_num",
        when(
            trim(col("buyers")).rlike("^[0-9]+$"),
            trim(col("buyers")).cast("int")
        ).otherwise(lit(0))
    )
    .filter(col("deploy_date").isNotNull())
)

buyers_df = (
    campaign_buyers_df
    .groupBy("deploy_date")
    .agg(
        spark_sum("buyers_num").alias("buyers")
    )
    .withColumnRenamed("deploy_date", "date")
)

# --------------------------------------------------
# FINAL CONSOLIDATION (NOTHING DROPPED)
# --------------------------------------------------
consolidated_df = (
    date_df
    .join(new_customer_df, "date", "left")
    .join(repeat_customer_df, "date", "left")
    .join(new_customer_sales_df, "date", "left")
    .join(repeat_customer_sales_df, "date", "left")
    .join(birthday_customer_df, "date", "left")
    .join(anniversary_customer_df, "date", "left")
    .join(birthday_customer_sales_df, "date", "left")
    .join(anniversary_customer_sales_df, "date", "left")
    .join(total_customer_txn_df, "date", "left")
    .join(transacted_customer_sales_df, "date", "left")
    .join(customer_birthdays_perday_df, "date", "left")
    .join(customer_anniversaries_perday_df, "date", "left")
    .join(daily_enrolled_df, "date", "left")
    .join(members_archived_df, "date", "left")
    .join(cross_channel_buyers_df, "date", "left")
    .join(diamond_enthusiasts_df, "date", "left")
    .join(dormant_customers_df, "date", "left")
    .join(low_point_balance_df, "date", "left")
    .join(active_customers_df, "date", "left")
    .join(campaigns_deployed_df, "date", "left")
    .join(targeted_count_df, "date", "left")
    .join(buyers_df, "date", "left")
    .fillna({
        "newcustomercount": 0,
        "repeatcustomercount": 0,
        "newcustomersales": 0.0,
        "repeatcustomersales": 0.0,
        "birthdaycustomercount": 0,
        "anncustomercount": 0,
        "birthdaycustomersales": 0.0,
        "annvcustomersales": 0.0,
        "totalcustomertransactions": 0,
        "transactedcustomersales": 0.0,
        "customerbirthdaysperday": 0,
        "customeranniversariesperday": 0,
        "enrolledcustomercount": 0,
        "membersarchived": 0,
        "crosschannelbuyers": 0,
        "diamondenthusiasts": 0,
        "dormantcustomers": 0,
        "lowpointbalancecustomers": 0,
        "activecustomers": 0,
        "campaignsdeployed": 0,
        "targetedcount": 0,
        "buyers": 0
    })
)

# --------------------------------------------------
# FINAL ORDER (LOCKED)
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
        "annvcustomersales",
        "totalcustomertransactions",
        "transactedcustomersales",
        "customerbirthdaysperday",
        "customeranniversariesperday",
        "enrolledcustomercount",
        "membersarchived",
        "crosschannelbuyers",
        "diamondenthusiasts",
        "dormantcustomers",
        "lowpointbalancecustomers",
        "activecustomers",
        "campaignsdeployed",
        "targetedcount",
        "buyers"
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
