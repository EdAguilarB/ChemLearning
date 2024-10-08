
from data.rhcaa import rhcaa_diene
import streamlit as st
import plotly.graph_objects as go
import sys
import os
import torch
import time
import pandas as pd
from copy import deepcopy
from call_methods import make_network, create_loaders
from utils.utils_model import train_network, eval_network, generate_st_report


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def train_GNNet(opt, file) -> None:

    st.write(f'Initializing {opt.experiment_name} experiment...')

    # Set the device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the dataset
    data = rhcaa_diene(opt = opt, 
                           filename = opt.filename,
                           molcols = opt.mol_cols, 
                           root=opt.root,
                           file=file)

    if opt.split_type == 'tvt':
        max_outer = 2
        max_inner = 2
        TOT_RUNS = 1

    elif opt.split_type == 'cv':
        max_outer = 2
        max_inner = opt.folds
        TOT_RUNS = opt.folds-1

    elif opt.split_type == 'ncv':
        max_outer = opt.folds+1
        max_inner = opt.folds
        TOT_RUNS = opt.folds*(opt.folds-1)

    ncv_iterators = create_loaders(data, opt)

    results_all = pd.DataFrame()
    report_all = []

    model_params = []
    models = []
    models_names = []

    # Initiate the counter of the total runs
    counter = 0

    # Loop through the nested cross validation iterators
    # The outer loop is for the outer fold or test fold
    for outer in range(1, max_outer):
        # The inner loop is for the inner fold or validation fold
        for inner in range(1, max_inner):

            # Inner fold is incremented by 1 to avoid having same inner and outer fold number for logging purposes
            real_inner = inner +1 if outer <= inner else inner

            if opt.split_type == 'tvt':
                st.title(f"Train-Validation-Test Split Training Process")

            elif opt.split_type == 'cv' or opt.split_type == 'ncv':
                st.title(f"Outer Fold: {outer} | Inner Fold: {real_inner} Training Process")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Train Loss', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Validation Loss', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Test Loss', line=dict(color='green')))
            fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Loss",
                title="Learning Curve"
            )
            plot_placeholder = st.empty()

            # Initiate the early stopping parameters
            val_best_loss = 1000
            early_stopping_counter = 0
            best_epoch = 0

            # Increment the counter
            counter += 1
            # Get the data loaders
            train_loader, val_loader, test_loader = next(ncv_iterators)

            # Initiate the lists to store the losses
            train_list, val_list, test_list = [], [], []

            # Create the GNN model
            model = make_network(network_name = opt.network_name,
                                 opt = opt, 
                                 n_node_features= data.num_node_features).to(device)
            
            # Start the timer for the training
            start_time = time.time()

            for epoch in range(opt.epochs):
                # Checks if the early stopping counter is less than the early stopping parameter
                if early_stopping_counter <= opt.early_stopping:
                    # Train the model
                    train_loss = train_network(model, train_loader, device)
                    # Evaluate the model
                    val_loss = eval_network(model, val_loader, device)
                    test_loss = eval_network(model, test_loader, device)  

                    print('{}/{}-Epoch {:03d} | Train loss: {:.3f} {} | Validation loss: {:.3f} {} | '             
                        'Test loss: {:.3f} {}'.format(counter, TOT_RUNS, epoch, train_loss, opt.target_variable_units, 
                                                          val_loss, opt.target_variable_units, test_loss, opt.target_variable_units))
                    
                    # Model performance is evaluated every 5 epochs
                    if epoch % 5 == 0:
                        # Scheduler step
                        model.scheduler.step(val_loss)
                        # Append the losses to the lists
                        train_list.append(train_loss)
                        val_list.append(val_loss)
                        test_list.append(test_loss)

                        fig.data[0].x = list(range(0, epoch, 5))
                        fig.data[0].y = train_list
                        fig.data[1].x = list(range(0, epoch, 5))
                        fig.data[1].y = val_list
                        fig.data[2].x = list(range(0, epoch, 5))
                        fig.data[2].y = test_list

                        plot_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        # Save the model if the validation loss is the best
                        if val_loss < val_best_loss:
                            # Best validation loss and early stopping counter updated
                            val_best_loss, best_epoch = val_loss, epoch
                            early_stopping_counter = 0
                            print('New best validation loss: {:.4f} found at epoch {}'.format(val_best_loss, best_epoch))
                            # Save the  best model parameters
                            best_model_params = deepcopy(model.state_dict())

                        else:
                            # Early stopping counter is incremented
                            early_stopping_counter += 1


                        # Remove the old vertical line trace if it exists
                        if len(fig.data) > 3:
                            fig.data = fig.data[:3]

                        # Update the vertical line
                        fig.add_trace(go.Scatter(
                            x=[best_epoch, best_epoch],
                            y=[0, max(*train_list, *val_list, *test_list)],
                            mode='lines',
                            name='Best Validation Loss (Saved Model)',
                            line=dict(color="red", width=2, dash="dashdot"),
                            hoverinfo='skip'
                        ))

                    if epoch == opt.epochs:
                        print('Maximum number of epochs reached')

                else:
                    print('Early stopping limit reached')
                    break
            
            print('---------------------------------')
            # End the timer for the training
            training_time = (time.time() - start_time)/60
            print('Training time: {:.2f} minutes'.format(training_time))

            print(f"Training for test outer fold: {outer}, and validation inner fold: {real_inner} completed.")
            print(f"Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}")

            print('---------------------------------')
            
            # Generate the training report
            result_run, report = generate_st_report(opt = opt, 
                                            loaders = (train_loader, val_loader, test_loader), 
                                            model = model, 
                                            model_params = best_model_params,
                                            inner=real_inner,
                                            outer=outer,)
            
            results_all = pd.concat([results_all, result_run], axis=0)
            report_all.extend(report)

            model_params.append(best_model_params)
            models.append(model)
            models_names.append(f'Outer Fold {outer} - Inner Fold {real_inner}')
            
            del model, train_loader, val_loader, test_loader, train_list, val_list, test_list, best_model_params, best_epoch, fig
        
        print(f'All runs for outer test fold {outer} completed')


    st.title("Final Results")

    results_all = results_all.reset_index(drop=True)

    st.write("CSV file with all predictions and real values")


    if opt.split_type != 'tvt':

        st.write("CSV file with mean predictions and real values")

        mean_preds = results_all.groupby([opt.mol_id_col, 'set'], as_index=False).mean()
        mean_preds[['Inner_Fold', 'Outer_Fold']] = 'Mean'

        median_preds = results_all.groupby([opt.mol_id_col, 'set'], as_index=False).median()
        median_preds[['Inner_Fold', 'Outer_Fold']] = 'Median'

        results_all = pd.concat([results_all, mean_preds, median_preds], axis=0)

        #results_all = results_all.groupby([opt.mol_id_col, 'set'], as_index=False).mean()

        #results_all = results_all.drop(columns=['Inner_Fold', 'Outer_Fold'])

        results_all_plot = results_all.loc[results_all['Inner_Fold'] == 'Mean']

        title = 'Parity Plot Mean Predictions'

    else:
        results_all_plot = results_all
        title = 'Parity Plot'
    
    st.write(results_all)

    results_train = results_all_plot.loc[results_all_plot['set'] == 'Training']
    results_val = results_all_plot.loc[results_all_plot['set'] == 'Validation']
    results_test = results_all_plot.loc[results_all_plot['set'] == 'Test']

    if opt.problem_type == 'regression':

        fig = go.Figure()


        fig.add_trace(go.Scatter(x=results_train[f'real_{opt.target_variable_name}'],
                                    y=results_train[f'predicted_{opt.target_variable_name}'],
                                    mode='markers',
                                    name='Training Set',
                                    marker=dict(color='blue'),
                                    text=results_train[opt.mol_id_col],
                                    hoverinfo='text'
                                    ))
        
        fig.add_trace(go.Scatter(x=results_val[f'real_{opt.target_variable_name}'],
                                        y=results_val[f'predicted_{opt.target_variable_name}'],
                                        mode='markers',
                                        name='Validation Set',
                                        marker=dict(color='orange'),
                                        text=results_val[opt.mol_id_col],
                                        hoverinfo='text'
                                        ))
        
        fig.add_trace(go.Scatter(x=results_test[f'real_{opt.target_variable_name}'],
                                        y=results_test[f'predicted_{opt.target_variable_name}'],
                                        mode='markers',
                                        name='Test Set',
                                        marker=dict(color='green'),
                                        text=results_test[opt.mol_id_col],
                                        hoverinfo='text'
                                        ))

        fig.add_trace(go.Scatter(x=[min(results_train[f'real_{opt.target_variable_name}']), max(results_train[f'real_{opt.target_variable_name}'])],
                                    y=[min(results_train[f'real_{opt.target_variable_name}']), max(results_train[f'real_{opt.target_variable_name}'])],
                                    mode='lines',
                                    name='Parity Line',
                                    line=dict(color='red', dash='dash')))
        
        fig.update_layout(title=title,
                        xaxis_title=f'Real {opt.target_variable_name} {opt.target_variable_units}',
                        yaxis_title=f'Predicted {opt.target_variable_name} {opt.target_variable_units}',
                        showlegend=True,
                        )

        if opt.split_type == 'tvt' and opt.show_all:
            pass
        else:
            st.plotly_chart(fig, use_container_width=True)

    
    return model_params, models, report_all, results_all




    


