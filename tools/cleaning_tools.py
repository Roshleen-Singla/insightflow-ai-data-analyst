import pandas as pd
import re
import difflib
import json
from pathlib import Path

from langchain_core.tools import tool


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the CSV dataset."""

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    return df


# ============================================================
# 2. REMOVE USELESS COLUMNS
# ============================================================

def remove_useless_columns(df: pd.DataFrame):
    """
    Remove columns that are not useful for the current analysis.

    Returns:
        cleaned dataframe
        dictionary explaining which columns were removed
    """

    removed_columns = {}

    for column in df.columns:

        # Pandas-generated columns
        if column.startswith("Unnamed:"):

            removed_columns[column] = (
                "Pandas-generated/unnamed column"
            )

        # Promotion IDs are not being used in the current analysis
        elif column == "promotion-ids":

            removed_columns[column] = (
                "Promotion ID information excluded "
                "from the current analysis"
            )

    if removed_columns:

        df = df.drop(
            columns=list(removed_columns.keys())
        )

    return df, removed_columns


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df: pd.DataFrame):
    """
    Handle missing values.

    Numerical columns:
        Missing values -> median

    Text columns:
        Missing values -> 'Unknown'

    Returns:
        cleaned dataframe
        detailed missing-value report
    """

    missing_report = {}

    # --------------------------------------------------------
    # Numerical columns
    # --------------------------------------------------------

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numerical_columns:

        missing_count = int(
            df[column].isnull().sum()
        )

        if missing_count > 0:

            median_value = df[column].median()

            # Column was entirely null — median() is NaN, so fall back to 0
            fill_value = median_value if pd.notna(median_value) else 0

            df[column] = df[column].fillna(fill_value)

            missing_report[column] = {
                "missing_values": missing_count,
                "method": "median" if pd.notna(median_value) else "constant (all-null column)",
                "replacement_value": float(fill_value)
            }

    # --------------------------------------------------------
    # Categorical / text columns
    # --------------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:

        missing_count = int(
            df[column].isnull().sum()
        )

        if missing_count > 0:

            df[column] = df[column].fillna(
                "Unknown"
            )

            missing_report[column] = {
                "missing_values": missing_count,
                "method": "constant",
                "replacement_value": "Unknown"
            }

    total_missing_before = sum(
        item["missing_values"]
        for item in missing_report.values()
    )

    total_missing_after = int(
        df.isnull().sum().sum()
    )

    return (
        df,
        total_missing_before,
        total_missing_after,
        missing_report
    )


# ============================================================
# 4. CLEAN CITIES
# ============================================================

def clean_cities(df: pd.DataFrame):
    """
    Clean and standardize city names.

    Uses:
        1. Basic normalization
        2. Exact matching
        3. Alias matching
        4. Fuzzy matching
        5. OTHER bucket for uncertain values
    """

    # --------------------------------------------------------
    # Canonical cities
    # --------------------------------------------------------

    CANONICAL_CITIES = [
        "MUMBAI", "DELHI", "NEW DELHI", "BENGALURU", "HYDERABAD",
        "CHENNAI", "KOLKATA", "PUNE", "AHMEDABAD", "SURAT",
        "JAIPUR", "LUCKNOW", "KANPUR", "NAGPUR", "INDORE",
        "THANE", "BHOPAL", "VISAKHAPATNAM", "PATNA", "VADODARA",
        "GHAZIABAD", "LUDHIANA", "AGRA", "NASHIK", "FARIDABAD",
        "MEERUT", "RAJKOT", "KALYAN", "VARANASI", "SRINAGAR",
        "AURANGABAD", "DHANBAD", "AMRITSAR", "NAVI MUMBAI",
        "ALLAHABAD", "PRAYAGRAJ", "RANCHI", "HOWRAH", "COIMBATORE",
        "JABALPUR", "GWALIOR", "VIJAYAWADA", "JODHPUR", "MADURAI",
        "RAIPUR", "KOTA", "GUWAHATI", "CHANDIGARH", "SOLAPUR",
        "HUBLI", "TIRUCHIRAPPALLI", "TIRUPPUR", "MORADABAD",
        "MYSORE", "MYSURU", "GURGAON", "GURUGRAM", "ALIGARH",
        "JALANDHAR", "BHUBANESWAR", "SALEM", "WARANGAL", "GUNTUR",
        "BHIWANDI", "SAHARANPUR", "GORAKHPUR", "BIKANER", "AMRAVATI",
        "NOIDA", "GREATER NOIDA", "JAMSHEDPUR", "BHILAI", "CUTTACK",
        "FIROZABAD", "KOCHI", "ERNAKULAM", "NELLORE", "BHAVNAGAR",
        "DEHRADUN", "DURGAPUR", "ASANSOL", "ROURKELA", "NANDED",
        "KOLHAPUR", "AJMER", "AKOLA", "GULBARGA", "KALABURAGI",
        "JAMNAGAR", "UJJAIN", "LONI", "SILIGURI", "JHANSI",
        "ULHASNAGAR", "JAMMU", "SANGLI", "MANGALORE", "MANGALURU",
        "ERODE", "BELGAUM", "BELAGAVI", "KURNOOL", "TIRUNELVELI",
        "RAJAHMUNDRY", "MALEGAON", "GAYA", "UDAIPUR", "MAHESHTALA",
        "TIRUPATI", "DAVANAGERE", "KOZHIKODE", "CALICUT", "AKBARPUR",
        "BOKARO", "SOUTH DUMDUM", "BELLARY", "PATIALA", "GOPALPUR",
        "AGARTALA", "BHAGALPUR", "MUZAFFARNAGAR", "BHATPARA",
        "PANIHATI", "LATUR", "DHULE", "ROHTAK", "KORBA", "BHILWARA",
        "BERHAMPUR", "MUZAFFARPUR", "AHMEDNAGAR", "MATHURA", "KOLLAM",
        "AVADI", "KADAPA", "KAMARHATI", "SAMBALPUR", "BILASPUR",
        "SHAHJAHANPUR", "SATARA", "BIJAPUR", "VIJAYAPURA", "RAMPUR",
        "SHIVAMOGGA", "SHIMOGA", "CHANDRAPUR", "JUNAGADH", "THRISSUR",
        "ALWAR", "BARDHAMAN", "KULTI", "KAKINADA", "NIZAMABAD",
        "PARBHANI", "TUMKUR", "TUMAKURU", "KHAMMAM", "OZHUKARAI",
        "BIHAR SHARIF", "PANIPAT", "DARBHANGA", "BALLY", "AIZAWL",
        "DEWAS", "ICHALKARANJI", "KARNAL", "BATHINDA", "JALGAON",
        "EAST DELHI", "KALYAN-DOMBIVLI", "DOMBIVLI",
        "KALYAN DOMBIVLI", "BARASAT", "KIRARI SULEMAN NAGAR",
        "PURNIA", "SATNA", "MAU", "SONIPAT", "FARRUKHABAD",
        "SAGAR", "DURG", "IMPHAL", "RATLAM", "HAPUR", "ARRAH",
        "KARIMNAGAR", "ANANTAPUR", "ETAWAH", "AMBERNATH",
        "AMBARNATH", "NORTH DUMDUM", "BHARATPUR", "BEGUSARAI",
        "GANDHIDHAM", "BARANAGAR", "TIRUVOTTIYUR", "PUDUCHERRY",
        "PONDICHERRY", "SIKAR", "THOOTHUKUDI", "TUTICORIN", "REWA",
        "MIRZAPUR", "RAICHUR", "PALI", "RAMAGUNDAM", "HARIDWAR",
        "VIJAYANAGARAM", "VIZIANAGARAM", "KATNI", "SRIGANGANAGAR",
        "SIRSA", "DANAPUR", "SHIMLA", "PANCHKULA", "MOHALI",
        "ZIRAKPUR", "GANDHINAGAR", "ANAND", "NADIAD", "MEHSANA",
        "BHUJ", "PORBANDAR", "VAPI", "VALSAD", "NAVSARI", "BARDOLI",
        "GODHRA", "PATAN", "MORBI", "VERAVAL", "PALANPUR", "SILVASSA",
        "DAMAN", "PANAJI", "MARGAO", "VASCO DA GAMA", "SHILLONG",
        "ITANAGAR", "KOHIMA", "GANGTOK", "DIBRUGARH", "JORHAT",
        "SILCHAR", "TEZPUR", "NAGAON", "TRIVANDRUM",
        "THIRUVANANTHAPURAM", "KOTTAYAM", "ALAPPUZHA", "PALAKKAD",
        "MALAPPURAM", "KANNUR", "PATHANAMTHITTA", "IDUKKI",
        "WAYANAD", "KASARAGOD", "PIMPRI", "CHINCHWAD",
        "PIMPRI-CHINCHWAD", "VASAI", "VIRAR", "VASAI-VIRAR",
        "BADLAPUR", "PANVEL", "KARAD"
    ]

    # --------------------------------------------------------
    # Aliases
    # --------------------------------------------------------

    ALIASES = {
        "BANGALORE": "BENGALURU",
        "BANGLORE": "BENGALURU",
        "BOMBAY": "MUMBAI",
        "CALCUTTA": "KOLKATA",
        "MADRAS": "CHENNAI",

        "GURGAON": "GURUGRAM",
        "MYSORE": "MYSURU",
        "MANGALORE": "MANGALURU",

        "CALICUT": "KOZHIKODE",
        "TRIVANDRUM": "THIRUVANANTHAPURAM",

        "PONDICHERRY": "PUDUCHERRY",
        "TUTICORIN": "THOOTHUKUDI",

        "SHIMOGA": "SHIVAMOGGA",
        "TUMKUR": "TUMAKURU",

        "BELGAUM": "BELAGAVI",
        "BIJAPUR": "VIJAYAPURA",
        "GULBARGA": "KALABURAGI",

        "ALLAHABAD": "PRAYAGRAJ",

        "VIJAYANAGARAM": "VIZIANAGARAM",

        "NEW DELHI": "DELHI",
        "EAST DELHI": "DELHI",

        "DOMBIVLI": "KALYAN-DOMBIVLI",
        "KALYAN DOMBIVLI": "KALYAN-DOMBIVLI",

        "PIMPRI": "PIMPRI-CHINCHWAD",
        "CHINCHWAD": "PIMPRI-CHINCHWAD",

        "VASAI": "VASAI-VIRAR",
        "VIRAR": "VASAI-VIRAR",

        "ERNAKULAM": "KOCHI",

        "AMBERNATH": "AMBARNATH"
    }

    # --------------------------------------------------------
    # Lookup
    # --------------------------------------------------------

    CANONICAL_SET = (
        set(CANONICAL_CITIES)
        | set(ALIASES.values())
    )

    EXACT_LOOKUP = {
        city: city
        for city in CANONICAL_SET
    }

    EXACT_LOOKUP.update(ALIASES)

    # --------------------------------------------------------
    # Basic city normalization
    # --------------------------------------------------------

    def clean_city_text(raw):

        if not isinstance(raw, str):
            return ""

        text = raw.upper()

        # Remove parenthetical information
        text = re.sub(
            r"\(.*?\)",
            " ",
            text
        )

        # Remove numbers / PIN codes
        text = re.sub(
            r"\d+",
            " ",
            text
        )

        # Replace punctuation except comma
        text = re.sub(
            r"[.\-/]",
            " ",
            text
        )

        # Collapse spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    # --------------------------------------------------------
    # City matcher
    # --------------------------------------------------------

    def match_city(raw, cutoff=0.90, min_len=5):

        cleaned = clean_city_text(raw)

        if not cleaned:
            return "OTHER", "empty"

        candidates = [
            part.strip()
            for part in cleaned.split(",")
            if part.strip()
        ]

        if cleaned not in candidates:
            candidates.append(cleaned)

        # ----------------------------------------------------
        # Exact / alias matching
        # ----------------------------------------------------

        for part in candidates:

            if part in EXACT_LOOKUP:

                if part in ALIASES:
                    return EXACT_LOOKUP[part], "alias"

                return EXACT_LOOKUP[part], "exact"

        # ----------------------------------------------------
        # Fuzzy matching
        # ----------------------------------------------------

        best_match = None
        best_score = 0.0

        for part in candidates:

            words = part.split()

            for word in words:

                if len(word) < min_len:
                    continue

                matches = difflib.get_close_matches(
                    word,
                    EXACT_LOOKUP.keys(),
                    n=1,
                    cutoff=cutoff
                )

                if matches:

                    matched_city = matches[0]

                    score = difflib.SequenceMatcher(
                        None,
                        word,
                        matched_city
                    ).ratio()

                    if score > best_score:

                        best_match = (
                            EXACT_LOOKUP[matched_city]
                        )

                        best_score = score

        if best_match:

            return best_match, "fuzzy"

        return "OTHER", "other"

    # --------------------------------------------------------
    # Check column exists
    # --------------------------------------------------------

    if "ship-city" not in df.columns:

        return df, {
            "error": "ship-city column not found"
        }

    # --------------------------------------------------------
    # Original statistics
    # --------------------------------------------------------

    original_unique = int(
        df["ship-city"].nunique()
    )

    # --------------------------------------------------------
    # Apply city matching
    # --------------------------------------------------------

    city_results = df["ship-city"].apply(
        match_city
    )

    df["ship-city-clean"] = city_results.apply(
        lambda x: x[0]
    )

    # Temporary audit column
    df["_city_match_method"] = city_results.apply(
        lambda x: x[1]
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    cleaned_unique = int(
        df["ship-city-clean"].nunique()
    )

    match_counts = (
        df["_city_match_method"]
        .value_counts()
        .to_dict()
    )

    other_count = int(
        (
            df["ship-city-clean"] == "OTHER"
        ).sum()
    )

    other_share = (
        other_count / len(df)
        if len(df) > 0
        else 0
    )

    # --------------------------------------------------------
    # City report
    # --------------------------------------------------------

    report = {

        "original_unique_cities": original_unique,

        "cleaned_unique_cities": cleaned_unique,

        "exact_matches": int(
            match_counts.get("exact", 0)
        ),

        "alias_matches": int(
            match_counts.get("alias", 0)
        ),

        "fuzzy_matches": int(
            match_counts.get("fuzzy", 0)
        ),

        "empty_values": int(
            match_counts.get("empty", 0)
        ),

        "cities_bucketed_as_other": other_count,

        "other_share": round(
            float(other_share) * 100,
            2
        )
    }

    # Remove temporary audit column
    df = df.drop(
        columns=["_city_match_method"]
    )

    return df, report


# ============================================================
# 5. CLEAN STATES
# ============================================================

def clean_states(df: pd.DataFrame):
    """
    Standardize state names using explicit mappings.

    Returns:
        cleaned dataframe
        detailed state correction report
    """

    STATE_MAPPING = {

        "RJ": "RAJASTHAN",

        "RAJSHTHAN": "RAJASTHAN",

        "RAJSTHAN": "RAJASTHAN",

        "PB": "PUNJAB",

        "ORISSA": "ODISHA",

        "PONDICHERRY": "PUDUCHERRY",

        "NL": "NAGALAND",

        "AR": "ARUNACHAL PRADESH",

        "PUNJAB/MOHALI/ZIRAKPUR": "PUNJAB"
    }

    if "ship-state" not in df.columns:

        return df, {
            "error": "ship-state column not found"
        }

    # Standardize formatting
    df["ship-state"] = (
        df["ship-state"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    correction_report = {}

    # --------------------------------------------------------
    # Apply only corrections that actually occur
    # --------------------------------------------------------

    for old_value, new_value in STATE_MAPPING.items():

        count = int(
            (
                df["ship-state"] == old_value
            ).sum()
        )

        if count > 0:

            correction_report[old_value] = {

                "changed_to": new_value,

                "rows_affected": count
            }

            df["ship-state"] = (
                df["ship-state"]
                .replace(
                    old_value,
                    new_value
                )
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_corrected = sum(
        item["rows_affected"]
        for item in correction_report.values()
    )

    report = {

        "corrections_applied": correction_report,

        "total_rows_corrected": total_corrected,

        "unique_states_after_cleaning": int(
            df["ship-state"].nunique()
        )
    }

    return df, report

# ============================================================
# 6. RENAME COLUMNS
# ============================================================
# Renaming columns to consistent, readable names is important for data analysis and reporting.
# The `rename_columns` function takes a DataFrame as input and renames its columns based on a predefined mapping. 
# It ensures that only existing columns are renamed, preventing errors from attempting to rename non-existent columns. 
# The function returns the modified DataFrame along with a dictionary of the columns that were renamed, providing transparency about the changes made.
def rename_columns(df: pd.DataFrame):
    """Rename columns to consistent, readable names."""

    column_mapping = {
          "index" : "index",
          "Order ID": "order_id",
          "Date": "order_date",
          "Status": "status",
          "Fulfilment": "fulfilment",
          "Sales Channel ": "sales_channel",
          "ship-service-level" : "ship_service_level",
          "Amount": "amount",
          "ship-service-level": "ship_service_level",
          "ship-postal-code": "ship_postal_code",
          "ship-city-clean": "ship_city",
          "ship-state": "ship_state",
          "ship-country": "ship_country",
          "Qty": "quantity",
          "Style": "style",
          "SKU": "sku",
          "ASIN" : "asin",
          "Category": "category",
          "Size": "size",
          "B2B": "b2b",
          "Courier Status": "courier_status",
          "fulfilled-by": "fulfilled_by"
    }

    columns_to_drop = ["ship-city"]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    existing_mapping = {
        old: new
        for old, new in column_mapping.items()
        if old in df.columns
    }
    
    df = df.rename(columns=existing_mapping)

    return df, existing_mapping


# ============================================================
# 7. SAVE CLEANED DATASET
# ============================================================

def save_cleaned_dataset(
    df: pd.DataFrame,
    output_path: str
):
    """Save cleaned dataset to CSV."""

    output_file = Path(output_path)

    # Create folder if needed
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_file,
        index=False
    )

    return str(output_file)


# ============================================================
# 8. COMPLETE CLEANING PIPELINE
# ============================================================

def run_cleaning_pipeline(
    input_path: str,
    output_path: str
):
    """
    Execute the complete cleaning pipeline.

    Steps:
        1. Load dataset
        2. Remove useless columns
        3. Handle missing values
        4. Clean cities
        5. Clean states
        6. Rename columns
        7. Save dataset
        8. Return report
    """

    # ========================================================
    # STEP 1 — LOAD
    # ========================================================

    df = load_dataset(
        input_path
    )

    original_rows = int(
        df.shape[0]
    )

    original_columns = int(
        df.shape[1]
    )

    # ========================================================
    # STEP 2 — REMOVE USELESS COLUMNS
    # ========================================================

    df, removed_columns = (
        remove_useless_columns(df)
    )

    # ========================================================
    # STEP 3 — HANDLE MISSING VALUES
    # ========================================================

    (
        df,
        missing_before,
        missing_after,
        missing_report
    ) = handle_missing_values(df)

    # ========================================================
    # STEP 4 — CLEAN CITIES
    # ========================================================

    df, city_report = clean_cities(df)

    # ========================================================
    # STEP 5 — CLEAN STATES
    # ========================================================

    df, state_report = clean_states(df)

    # ========================================================
    # STEP 6 — RENAME COLUMNS
    # ========================================================
    df, rename_report = rename_columns(df)

    # ========================================================
    # STEP 7 — SAVE
    # ========================================================

    saved_path = save_cleaned_dataset(
        df,
        output_path
    )

    # ========================================================
    # STEP 8 — FINAL REPORT
    # ========================================================

    report = {

        "dataset": {

            "original_rows": original_rows,

            "original_columns": original_columns,

            "final_rows": int(
                df.shape[0]
            ),

            "final_columns": int(
                df.shape[1]
            )
        },

        "columns_removed": removed_columns,

        "missing_values": {

            "total_before": missing_before,

            "total_after": missing_after,

            "details": missing_report
        },

        "city_cleaning": city_report,

        "state_cleaning": state_report,

        "renamed_columns": rename_report,

        "output": {

            "saved_to": saved_path
        }
    }

    return df, report


# ============================================================
# AI TOOL
# ============================================================

@tool
def clean_tool(
    input_path: str,
    output_path: str = "data/cleaned_amazon_sales.csv"
) -> str:
    """
    Cleans the Amazon sales dataset.

    The tool:
    1. Loads the dataset
    2. Removes useless columns
    3. Handles missing values
    4. Cleans and standardizes city names
    5. Cleans and standardizes state names
    6. Renames columns to consistent, readable names
    7. Saves the cleaned dataset
    8. Returns a detailed cleaning report
    """

    _, report = run_cleaning_pipeline(
        input_path,
        output_path
    )

    return json.dumps(
        report,
        indent=2,
        default=str
    )