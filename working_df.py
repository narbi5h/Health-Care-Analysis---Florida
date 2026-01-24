

'''************************************* DB CONNECTION *********************************'''

def filtered_df ():# 1) Connect to DB  (edit credentials as needed)

    import pandas as pd
    import os
    from dotenv import load_dotenv
    from sqlalchemy import create_engine, text

    #### Database Connection Setup

    # Load credentials (like DB_USER, DB_PASS, etc.) from the .env file
    load_dotenv()

    # Retrieve database credentials from environment variables
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    # Build the PostgreSQL connection string dynamically using credentials
    DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Initialize the SQLAlchemy engine to manage the connection to the database
    # 'pool_pre_ping=True' ensures stale connections are checked before use
    engine = create_engine(DB_URL, pool_pre_ping=True)

    # Confirm successful connection
    print("The connection to the DB has been successfully established ✅")


    # Load your zip/county file
    zips = pd.read_csv("../dimensions/fl_selected_counties_zipcodes.csv")[["zip_code", "county"]]


    # Create a temporary table and load the zip/county data into it
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TEMP TABLE zips (
                zip_code text,
                county   text
            );
        """))
        zips.to_sql("zips", conn, if_exists="append", index=False)

    # 2) Query the data you need
    query =f"""SELECT
                s.*,
                z.county
            FROM standardized_cpt_columns s
            JOIN zips z
                ON LEFT(s."ZIP4"::text, 5) = LEFT(z.zip_code::text, 5)
            WHERE
                (s."standard_charge_negotiated_dollar" IS NOT NULL
                OR s."standard_charge_negotiated_percentage" IS NOT NULL)
                AND s."specification" IN (
                    'United Health Care',
                    'Cigna',
                    'Aetna',
                    'Blue Cross Blue Shield',
                    'Medicare',
                    'Medicaid'
                )
                AND s."bucket" in ('Government', 'Commercial')
    """

    with engine.connect() as conn:
        result = conn.execute(text(query))   # the big query
        rows = result.fetchall()             # get all rows
        cols = result.keys()                 # column names

    working_df = pd.DataFrame(rows, columns=cols)
    working_df["ZIP4"] = working_df["ZIP4"].astype(str).str[:5]
    # normalize column names to lowercase
    working_df.columns = working_df.columns.str.lower()


    print("Rows loaded:", len(working_df))
    print("Columns:", working_df.columns.tolist())
    # working_df.head()

    return working_df



