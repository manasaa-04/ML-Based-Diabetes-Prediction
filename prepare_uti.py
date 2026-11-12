import pandas as pd

print("Loading UTI dataset...")

df = pd.read_excel("UTIEntryModel2017to2018Dataset.xlsx")

# Select female records only
female_df = df[df["GENDER"] == "F"].copy()

# Save female-only dataset
female_df.to_csv("uti_female.csv", index=False)

print("\nFemale dataset prepared successfully!")
print("Total female records:", len(female_df))

print("\nUTI target distribution:")
print(female_df["TARGET_UTI_ENTRY"].value_counts())

print("\nOriginal data partitions:")
print(female_df["partition"].value_counts())

print("\nUTI outcomes by partition:")
print(
    pd.crosstab(
        female_df["partition"],
        female_df["TARGET_UTI_ENTRY"]
    )
)

print("\nSaved as uti_female.csv")