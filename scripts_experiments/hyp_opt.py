import streamlit as st
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import os
import shutil
from ray import tune
from utils.hyp_opt_utils import train_model_ray
from data.rhcaa import rhcaa_diene



def run_tune(opt, config, file):

    # config is for tuneable hyperparameters
    # opt is for fixed hyperparameters


    log_dir = f"{opt.experiment_name}/{opt.filename[:-4]}/{opt.network_name}/results_hyp_opt"

    os.makedirs(log_dir, exist_ok=True)

    print(opt.root)


    # generates the graph if they don't exist
    _ = rhcaa_diene(opt=opt, 
                    filename=opt.filename, 
                    molcols=opt.mol_cols,
                    root=f'{opt.root}', 
                    file=file)



    num_samples = 5

    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric="test_loss",
        mode="min",
        max_t=300,
        grace_period=50,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=["lr", "n_convolutions", "embedding_dim", "readout_layers", "batch_size"],
        metric_columns=["val_loss", "test_loss", "training_iteration"]
    )

    st.write("Starting hyperparameter optimization...")
    st.write("This may take a while. Please be patient.")

    result = tune.run(
        tune.with_parameters(train_model_ray, opt=opt, file=file),
        resources_per_trial={"cpu": 1, "gpu": 0.},  # Adjust based on your resources
        config=config,
        num_samples=num_samples,  # Number of hyperparameter combinations to try
        scheduler=scheduler,
        progress_reporter=reporter,
        storage_path=f"{log_dir}/ray_results",
        name="tune_hyperparameters",
    )

    best_trial = result.get_best_trial("test_loss", "min", "last")
    best_config = best_trial.config
    best_test_loss = best_trial.last_result["test_loss"]

    st.write("Best trial config: {}".format(best_config))
    st.write("Best trial final test loss: {:.4f}".format(best_test_loss))


    all_runs_df = result.results_df

    shutil.rmtree(log_dir)

    return best_config, all_runs_df



