import os
import pandas as pd


def save_to_excel(results, filename):
    # Создаём папку если её нет
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df = pd.DataFrame(results)
    numeric_cols = df.select_dtypes(include='number').columns
    averages = df[numeric_cols].mean().to_dict()
    avg_row = {col: averages.get(col, "") for col in df.columns}
    avg_row["Context (EN)"] = "СРЕДНЕЕ ЗНАЧЕНИЕ"
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    df.to_excel(filename, index=False, engine="openpyxl")
    print(f"💾 Результаты сохранены в {filename}")
