import argparse
import pprint
from pathlib import Path

import optuna
import pandas as pd

pp = pprint.PrettyPrinter(width=60, indent=4)

parser = argparse.ArgumentParser()
parser.add_argument("--storage", type=str, required=True, help="Path to SQL-lite db")
parser.add_argument("--study_name", type=str, required=True, help="Name of the study")
parser.add_argument(
    "--run_id", type=str, required=False, help="Path to directory where run logs are stored"
)
args = parser.parse_args()

study = optuna.load_study(study_name=args.study_name, storage=f"sqlite:///{args.storage}")
pp.pprint(f"Number of trials: {len(study.trials)}")
pp.pprint(f"Best trial:  {study.best_trial.number}")
pp.pprint(f"Best value:  {study.best_value}")
pp.pprint(study.best_params)


if args.run_id:
    files = sorted(Path(args.run_id).glob("**/emissions.csv"))
    emissions = pd.concat([pd.read_csv(f) for f in files]).reset_index()

    kwh = emissions.energy_consumed.sum()
    kgco2 = emissions.emissions.sum()
    kgco2_kwh = emissions.emissions_rate.mean()

    print("Energy consumed: ", kwh, "kW/h")
    print("Average Emission rate: ", kgco2_kwh, "kg CO2/kWh")
    print("CO2 produced: ", kgco2, "kgCO2 equivalent")
