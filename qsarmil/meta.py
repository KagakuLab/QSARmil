from __future__ import annotations

import pandas as pd
from qsarcons.consensus import SystematicSearch, GeneticSearch
from rdkit import RDLogger
from sklearn.model_selection import train_test_split

from qsarmil.lazy import LazyMIL

RDLogger.DisableLog("rdApp.*")


class MultiConformerModel:
    """End-to-end lazy MIL pipeline with automatic model selection.

    Wraps :class:`~qsarmil.lazy.LazyMIL` to train every built-in
    descriptor/estimator combination, then picks the best-performing
    consensus of models on the validation split via a genetic search
    (:class:`qsarcons.consensus.GeneticSearch`) and applies it to the test set.
    """

    def __init__(
        self,
        num_conf: int = 10,
        hopt: bool = False,
        num_cpu: int = 20,
        output_folder: str | None = None,
        verbose: bool = True,
    ) -> None:
        """Store the settings passed through to the underlying LazyMIL run.

        Args:
            num_conf (int): Number of conformers to generate per molecule.
            hopt (bool): Whether to hyperparameter-tune each estimator.
            num_cpu (int): Number of CPU threads to use for conformer generation.
            output_folder (str): Directory for LazyMIL's intermediate
                prediction CSVs. Must be a real path (see
                :class:`~qsarmil.lazy.LazyMIL`).
            verbose (bool): Whether to print progress from the underlying steps.
        """
        super().__init__()

        self.num_conf = num_conf
        self.num_cpu = num_cpu
        self.hopt = hopt
        self.output_folder = output_folder
        self.verbose = verbose

    def run_predict(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
        """Train, select the best model consensus, and predict on the test set.

        Splits off a validation set from ``df_train``, trains every
        descriptor/estimator combination via :class:`~qsarmil.lazy.LazyMIL`,
        runs a genetic search over the validation predictions to pick the
        best-performing consensus of models, and applies that consensus to
        the test predictions.

        Args:
            df_train (pd.DataFrame): Training data; column 0 is SMILES,
                column 1 is the target.
            df_test (pd.DataFrame): Data to predict on; column 0 is SMILES.
                A target column is not required and is filled with ``None``
                if missing.

        Returns:
            pd.DataFrame: Two columns, ``SMILES`` and ``pred``, one row per
            test molecule.
        """

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

        cons_search = GeneticSearch(cons_size="auto", n_iter=50)

        best_cons = cons_search.run(x_val, true_val)
        pred_test = cons_search.predict(x_test[best_cons])

        # 6. Return predictions with df
        pred_df = pd.concat([res_test["SMILES"], pd.Series(pred_test)], axis=1)
        pred_df = pred_df.rename(columns={0: "pred"})

        if self.verbose:
            print(f"Best consensus:")
            print("\n".join(best_cons))

        return pred_df
