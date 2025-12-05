import os
import pandas as pd

def main():
    # Use raw string (r"...") to avoid Windows escape issues
    input_csv = r"C:\Internship\graph_rag_GM\data\Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv"

    # Output folder
    output_folder = "input"
    os.makedirs(output_folder, exist_ok=True)

    # Output CSV path
    output_csv = os.path.join(output_folder, "processed_data.csv")

    print(f"Reading from: {input_csv}")
    df = pd.read_csv(input_csv)

    # Drop Image_Name column (case-insensitive)
    drop_cols = [col for col in df.columns if col.lower() == "image_name"]
    if drop_cols:
        print(f"Dropping column: {drop_cols[0]}")
        df = df.drop(columns=drop_cols)

    # ---- NEW: Create merged text column ----
    # Find matching columns regardless of capitalization
    def find_col(name):
        for c in df.columns:
            if c.lower() == name.lower():
                return c
        raise KeyError(f"Column '{name}' not found in CSV. Available: {df.columns.tolist()}")

    title_col = find_col("Title")
    ingredients_col = find_col("Ingredients")
    instructions_col = find_col("Instructions")

    df["merged_text"] = (
        df[title_col].fillna("") +
        "\n\nIngredients:\n" + df[ingredients_col].fillna("") +
        "\n\nInstructions:\n" + df[instructions_col].fillna("")
    )

    # ----------------------------------------

    # Save to input folder
    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned file with merged_text column: {output_csv}")

if __name__ == "__main__":
    main()
