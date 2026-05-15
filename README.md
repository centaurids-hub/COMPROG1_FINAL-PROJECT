# Aquaculture Water Quality Dataset – AQU-01

## 👤 Student Information
- **Name:** Jesier Cresencio  
- **Student ID:** [25-2043]  
- **Course/Section:** BSECE 1C  
- **Professor:** [Engr. Gilfred Allen Madrigal]  
- **School:** Technological University of the Philippines – Manila  

---

## 📊 Project Title
**Anoxic Event Prediction using Aquaculture Water Quality Data**

---

## 📁 Dataset Description
This dataset contains water quality parameters collected from aquaculture ponds, including:
- Temperature (TEMP)
- Dissolved Oxygen (DO)
- pH
- Ammonia
- Nitrate
- Turbidity and other sensor readings

The data is used to analyze environmental conditions that may lead to anoxic events (low oxygen levels).

---

## ⚙️ Unique Filter Logic
To ensure a unique and relevant dataset, the following filter was applied:

- Data is limited to **station1**
- Only records from the **month of June (Month = 6)** are included
- **Dissolved Oxygen (DO < 6.5 mg/L)** is used to isolate low-oxygen conditions

June was selected because it has the **highest average temperature** in the dataset. Higher temperatures reduce oxygen solubility in water, making it an ideal condition for analyzing potential anoxic events.

---

## 🎯 Purpose
The filtered dataset focuses on environmental conditions where anoxic events are more likely to occur. This supports data analysis, visualization, and machine learning tasks for predicting low dissolved oxygen scenarios.

---

## 📌 Notes
- Original dataset is stored as `Aquapond Dataset. csv`
- Filtered dataset is stored as `filtered_dataset.csv`
- Data processing and filtering are implemented using Python (pandas & NumPy)
