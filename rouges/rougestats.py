import pandas as pd
import matplotlib.pyplot as plt

csv_df = pd.read_csv("./rouges/catest_PLarge0.csv")

r1 = csv_df["rouge1"].describe().reset_index()["rouge1"]
r2 = csv_df["rouge2"].describe().reset_index()["rouge2"]
rl = csv_df["rougeL"].describe().reset_index()["rougeL"]

summary = pd.DataFrame({"PLarge0 CA_TEST R1": list(r1), 
                        "PLarge0 CA_TEST R2": list(r2), 
                        "PLarge0 CA_TEST RL": list(rl)},
                        index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"])

print(summary.to_string())

'''
RESULTS OF PEGASUS_BILLSUM ON STUFF

pegasus_df = pd.read_csv("pegasussumms_rouges.csv")
extract_df = pd.read_csv("extractivesumms_rouges.csv")
catest_df = pd.read_csv("catest_rouges.csv")

pr1 = pegasus_df["rouge1"].describe().reset_index()["rouge1"]
pr2 = pegasus_df["rouge2"].describe().reset_index()["rouge2"]
prL = pegasus_df["rougeL"].describe().reset_index()["rougeL"]

er1 = extract_df["rouge1"].describe().reset_index()["rouge1"]
er2 = extract_df["rouge2"].describe().reset_index()["rouge2"]
erL = extract_df["rougeL"].describe().reset_index()["rougeL"]

car1 = catest_df["rouge1"].describe().reset_index()["rouge1"]
car2 = catest_df["rouge2"].describe().reset_index()["rouge2"]
carL = catest_df["rougeL"].describe().reset_index()["rougeL"]

summary = pd.DataFrame({"Pegasus R1": list(pr1), 
                        "Extract R1": list(er1), 
                        "CA R1": list(car1),

                        "Pegasus R2": list(pr2),
                        "Extract R2": list(er2), 
                        "CA R2": list(car2),

                        "Pegasus RL": list(prL), 
                        "Extract RL": list(erL), 
                        "CA RL": list(carL)},
                        index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"])

print(summary.to_string())
'''
