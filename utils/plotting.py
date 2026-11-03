import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_dataframe(df, scaling_dict = None , naming_dict = None ,save=False, path=None, show=False, single_col=False):
    first_column = df.columns[0]
    
    num_rows = int(np.ceil((len(df.columns)-1) / 2)) if not single_col else len(df.columns)-1
    # plt.rcParams.update({'font.size': 26})
    fig, axs = plt.subplots(nrows=num_rows, ncols=2 if not single_col else 1, figsize=(20, 7*num_rows))
    
    for i, column in enumerate(df.columns[1:], start=1):
        if single_col:
            row = i-1
            ax = axs[row]
        else:
            row = (i-1) // 2 if not single_col else i-1
            col = (i-1) % 2 if not single_col else 0
            ax = axs[row, col]

        ax.plot(df[first_column], df[column], linewidth=2)
        ax.set_xlabel(first_column)
        ax.set_ylabel(column)
        ax.set_title(f'{column}')
    
        # axs[row, col].xaxis.set_major_locator(ticker.MultipleLocator(10))
        # axs[row, col].xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(True) #, linewidth=0.5, linestyle='--', color='0.5')

    if scaling_dict is not None:
        if single_col:
            raise NotImplementedError("Custom scaling with scaling_dict not supported for single-column representation.")
        set_yaxis_scaling(df, axs, scaling_dict)
    if naming_dict is not None:
        set_subplot_titles_and_labels(df, axs, naming_dict, single_col)
    # If the number of plots is odd, remove the last (empty) subplot
    if not single_col and len(df.columns) % 2 == 0:
        fig.delaxes(axs[-1, -1])

    plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=3.0)
    if save:
        plt.savefig(path)
    if show:
        plt.show()


def plot_dataframe_comparison(df, ref_df, scaling_dict=None, naming_dict=None, save=False, path=None, show=False, single_col=False):
    first_column = df.columns[0]
    ref_first_column = ref_df.columns[0]
    
    num_rows = int(np.ceil((len(df.columns)-1) / 2)) if not single_col else len(df.columns)-1
    fig, axs = plt.subplots(nrows=num_rows, ncols=2 if not single_col else 1, figsize=(20, 7*num_rows))
    
    for i, column in enumerate(df.columns[1:], start=1):
        if single_col:
            row = i-1
            ax = axs[row]
        else:
            row = (i-1) // 2
            col = (i-1) % 2
            ax = axs[row, col] if num_rows > 1 else axs[col] # handles 1-row multi-col or multi-row grids safely

        # Plot the primary result line
        ax.plot(df[first_column], df[column], linewidth=2, label="Model Result", color="#b41f1f")
        
        # Plot the reference baseline line if the column exists in ref_df
        if column in ref_df.columns:
            ax.plot(ref_df[ref_first_column], ref_df[column], linewidth=1.5, linestyle='--', color="#696969", label="Reference Baseline")
        
        if first_column == 't':
            ax.set_xlabel('Time in s')
        else:
            ax.set_xlabel(first_column)

        ax.set_ylabel(column)
        ax.set_title(f'{column}')
        ax.grid(True)
        ax.legend(loc='best', frameon=False)

    if scaling_dict is not None:
        if single_col:
            raise NotImplementedError("Custom scaling with scaling_dict not supported for single-column representation.")
        set_yaxis_scaling(df, axs, scaling_dict)
        
    if naming_dict is not None:
        set_subplot_titles_and_labels(df, axs, naming_dict, single_col)
        
    # If the total number of subplots is odd (accounting for columns minus time), remove the last empty subplot
    if not single_col and (len(df.columns) - 1) % 2 != 0:
        # Safely handle indexing for 2D axes array
        flat_axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
        fig.delaxes(flat_axs[-1])

    plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=3.0)
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def plot_dataframe_individual_time_resolution(df_list, naming_dict = None ,save=False, path=None, show=False, single_col=False):
    plot_sum_plus1 = 0
    plot_cnt = 1

    for df in df_list:
        plot_sum_plus1 += len(df.columns)-1

    num_rows = int(np.ceil((plot_sum_plus1) / 2)) if not single_col else plot_sum_plus1
    plt.rcParams.update({'font.size': 26})
    fig, axs = plt.subplots(nrows=num_rows, ncols=2 if not single_col else 1, figsize=(20, 7*num_rows))

    for df in df_list:
        first_column = df.columns[0]

        for _, column in enumerate(df.columns[1:], start=1):
            i = plot_cnt

            if single_col:
                row = i-1
                ax = axs[row]
            else:
                row = (i-1) // 2 if not single_col else i-1
                col = (i-1) % 2 if not single_col else 0
                ax = axs[row, col]

            ax.plot(df[first_column], df[column], linewidth=2)
            ax.set_xlabel(first_column)
            ax.set_ylabel(column)
            ax.set_title(f'{column}')
        
            # axs[row, col].xaxis.set_major_locator(ticker.MultipleLocator(10))
            # axs[row, col].xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.grid(True) #, linewidth=0.5, linestyle='--', color='0.5')
            plot_cnt += 1

    if naming_dict is not None:
        set_subplot_titles_and_labels_split_dataframe(df_list, axs, naming_dict, single_col)

    # If the number of plots is odd, remove the last (empty) subplot
    if not single_col and plot_sum_plus1 % 2 == 0:
        fig.delaxes(axs[-1, -1])

    plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=3.0)
    if save:
        plt.savefig(path)
    if show:
        plt.show()

def set_yaxis_scaling(df, axs, scaling_dict):
    for i, column in enumerate(df.columns[1:], start=1):
        row = (i-1) // 2
        col = (i-1) % 2
        ax = axs[row, col]

        scaling_params = scaling_dict.get(column, {})
        if 'ylim' in scaling_params:
            ax.set_ylim(scaling_params['ylim'])
        if 'yscale' in scaling_params:
            ax.set_yscale(scaling_params['yscale'])

def set_subplot_titles_and_labels(df, axs, naming_dict, single_col, split_df=False, plot_cnt=-1):
    for j, column in enumerate(df.columns[1:], start=1):
        if split_df:
            i = plot_cnt
        else:
            i = j

        if single_col:
            row = i-1
            ax = axs[row]
        else:
            row = (i-1) // 2 if not single_col else i-1
            col = (i-1) % 2 if not single_col else 0
            ax = axs[row, col]

        title_params = naming_dict.get(column, {})
        if 'title' in title_params:
            ax.set_title(title_params['title'])
        if 'xlabel' in title_params:
            ax.set_xlabel(title_params['xlabel'])
        if 'ylabel' in title_params:
            ax.set_ylabel(title_params['ylabel'])

        if split_df:
            plot_cnt += 1

    return plot_cnt

def set_subplot_titles_and_labels_split_dataframe(df_list, acs, naming_dict, single_col):
    plot_cnt = 1
    for df in df_list:
        plot_cnt = set_subplot_titles_and_labels(df, acs, naming_dict, single_col, True, plot_cnt)

def plot_func(func):
    plt.figure()
    plt.plot(func, linewidth=2)
    plt.show()

def plot_dict(the_dict):
    lists = sorted(the_dict.items())
    plt.figure()
    x, y = zip(*lists)
    plt.plot(x, y, linewidth=2)
    plt.show()

def plot_multiple_lines(df):
    first_column = df.columns[0]
    plt.figure()
    plt.rcParams.update({'font.size': 26})

    for column in df.columns[1:]:
        plt.plot(df[first_column], df[column], label=column, linewidth=2)

    plt.legend()
    plt.show()


