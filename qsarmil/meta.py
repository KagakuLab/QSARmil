import pandas as pd
from qsarcons.consensus import SystematicSearch, GeneticSearch
from rdkit import RDLogger
from sklearn.model_selection import train_test_split

from qsarmil.lazy import LazyMIL

RDLogger.DisableLog("rdApp.*")


class MultiConformerModel:

    def __init__(self, num_conf=10, hopt=False, num_cpu=20, output_folder=None, verbose=True):
        super().__init__()

        self.num_conf = num_conf
        self.num_cpu = num_cpu
        self.hopt = hopt
        self.output_folder = output_folder
        self.verbose = verbose

    def run_predict(self, df_train, df_test):

        # 1. Fill fake test prop
        if len(df_test.columns) == 1:
            df_test[1] = [None for i in df_test.index]

        # 2. Train/val split
        df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)

        # 3. Build multiple models
        lazy_ml = LazyMIL(num_conf=self.num_conf, hopt=self.hopt,
                          num_cpu=self.num_cpu, output_folder=self.output_folder, verbose=self.verbose)
        lazy_ml.run(df_train, df_val, df_test)

        # 4. Load individual model predictions
        res_val = pd.read_csv(f"{self.output_folder}/val.csv")
        res_test = pd.read_csv(f"{self.output_folder}/test.csv")

        x_val, true_val = res_val.iloc[:, 2:], res_val.iloc[:, 1]
        x_test = res_test.iloc[:, 2:]

        # 5. Run genetic search
        if self.verbose:
            print("\nRunning genetic consensus search ...")

        cons_search = GeneticSearch(cons_size="auto", metric="auto", n_iter=50)
        # cons_search = SystematicSearch(cons_size="auto", metric="auto")

        best_cons = cons_search.run(x_val, true_val)
        pred_test = cons_search.predict(x_test[best_cons])

        # 6. Return predictions with df
        pred_df = pd.concat([res_test["SMILES"], pd.Series(pred_test)], axis=1)
        pred_df = pred_df.rename(columns={0: "pred"})

        if self.verbose:
            print(f"Best consensus:")
            print("\n".join(best_cons))

        return pred_df
