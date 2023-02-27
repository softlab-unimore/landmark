import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class PlotExplanation:
    @staticmethod
    def plot_impacts(data, target_col, ax, title):

        n = len(data)
        ax.set_xlim(-0.5, 0.5)  # set x axis limits
        ax.set_ylim(-1, n)  # set y axis limits
        ax.set_yticks(range(n))  # add 0-n ticks
        ax.set_yticklabels(data[['column', 'word']].astype(str).apply(lambda x: ', '.join(x), 1))  # add y tick labels

        # define arrows
        arrow_starts = np.repeat(0, n)
        arrow_lengths = data[target_col].values
        # add arrows to plot
        for i, subject in enumerate(data['column']):

            if subject.startswith('l'):
                arrow_color = '#347768'
            elif subject.startswith('r'):
                arrow_color = '#6B273D'
            else:
                raise ValueError("Subject doesn't start with 'l' or 'r' values.")

            if arrow_lengths[i] != 0:
                ax.arrow(arrow_starts[i],  # x start point
                         i,  # y start point
                         arrow_lengths[i],  # change in x
                         0,  # change in y
                         head_width=0,  # arrow head width
                         head_length=0,  # arrow head length
                         width=0.4,  # arrow stem width
                         fc=arrow_color,  # arrow fill color
                         ec=arrow_color,
                         )  # arrow edge color
                width = arrow_lengths[i]
                offset = 40
                offset = offset if width > 0 else -offset
                ax.annotate(format(width, '.3f'),
                        (width, i),
                        ha='right' if width > 0 else 'left',
                        va='center',
                        xytext=(offset, 2),
                        textcoords='offset points')

        # format plot
        ax.set_title(title)  # add title
        ax.axvline(x=0, color='0.9', ls='--', lw=2, zorder=0)  # add line at x=0
        ax.grid(axis='y', color='0.9')  # add a light grid
        ax.set_xlim(-0.5, 0.5)  # set x axis limits
        ax.set_xlabel('Token impact')  # label the x axis
        sns.despine(left=True, bottom=True, ax=ax)

    @staticmethod
    def plot_landmark(exp, landmark):

        if landmark == 'right':
            target_col = 'score_right_landmark'
        else:
            target_col = 'score_left_landmark'

        data = exp.copy()

        # sort individuals by amount of change, from largest to smallest
        data = data.sort_values(by=target_col, ascending=True) \
            .reset_index(drop=True)

        # initialize a plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))  # create figure

        if target_col == 'score_right_landmark':
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('l')], target_col, axes[0],
                                         'Original Tokens')
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[1],
                                         'Augmented Tokens')
        else:
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[0],
                                         'Original Tokens')
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('l')], target_col, axes[1],
                                         'Augmented Tokens')
            # fig.suptitle('Right Landmark Explanation')
        fig.tight_layout()

    @staticmethod
    def plot(exp, figsize=(16, 6)):
        data = exp.copy()

        data['column'] = data['column'].str.replace('left_','l_').str.replace('right_','r_')
        # initialize a plot
        if data[data['column'].str.startswith('r')]['score_right_landmark'].abs().max() > 0.01:
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize)  # create figure
            for target_col, land_side, ax in zip(['score_right_landmark', 'score_left_landmark'], ['Right', 'Left'], axes[[1,3]]):
                # sort individuals by amount of change, from largest to smallest
                side_char = land_side[0].lower()
                data = data.sort_values(by=target_col, ascending=True).reset_index(drop=True)
                PlotExplanation.plot_impacts(data[data['column'].str.startswith(side_char)], target_col, ax,
                                                 'Augmented Tokens')
            axes_for_original = axes[[0,2]]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)  # create figure
            axes_for_original = axes

        for target_col, land_side, ax in zip(['score_right_landmark', 'score_left_landmark'],['Right', 'Left'], axes_for_original):
            # sort individuals by amount of change, from largest to smallest
            side_char = land_side[0].lower()
            opposite_side_char = 'r' if side_char == 'l' else 'l'
            data = data.sort_values(by=target_col, ascending=True).reset_index(drop=True)
            PlotExplanation.plot_impacts(data[data['column'].str.startswith(opposite_side_char)], target_col, ax, 'Original Tokens')
            # if data[data['column'].str.startswith(side_char)][target_col].abs().max()>0.05:
            #     PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[1], 'Augmented Tokens')
            ax.set_ylabel(f'{land_side} Landmark')
        # axes[1].set_ylabel('Right Landmark')

        # target_col = 'score_left_landmark'
        # data = data.sort_values(by=target_col, ascending=True).reset_index(drop=True)
        # PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[2], 'Original Tokens')
        # if data[data['column'].str.startswith('l')][target_col].abs().max() > 0.05:
        #     PlotExplanation.plot_impacts(data[data['column'].str.startswith('l')], target_col, axes[3], 'Augmented Tokens')
        # axes[2].set_ylabel('Left Landmark')
        # # axes[3].set_ylabel('Left Landmark')

        fig.tight_layout()

        # plt.plot([0.5, 0.5], [0, 1], color='black', linestyle='--', lw=1, transform=gcf().transFigure, clip_on=False)
        # plt.plot([0, 1], [0.5, 0.5], color='lightgreen', lw=5, transform=gcf().transFigure, clip_on=False)

        return fig, axes

    @staticmethod
    def plot_counterfactual(data_df, pred_percentage=True):
        def generate_strikethrough_description(encoded_description, tokens_to_remove):
            new_description = ''
            whitespace = ' '

            for desc_token in encoded_description:
                token = desc_token.split('_')[-1]  # remove prefix
                if desc_token in tokens_to_remove:
                    new_description = whitespace.join([new_description, f'<del>{token}</del>'])
                else:
                    new_description = whitespace.join([new_description, token])

            return new_description

        html_page = '<html>'
        html_page += """<head>
            <style>
                table, tr, td, th {
                    border: 1px solid gray;
                    text-align: center;
                    border-collapse: collapse;
                }
                
                del {
                    color: red;
                }
                
                .tr2 {
                    border-bottom: 3px solid black;
                }
                
                .tr1 {
                    border-top: 3px solid black;
                }
                
            </style>
        </head>
        """
        html_page += '<body>'
        html_page += '<table>'
        html_page += """
            <tr>
                <th></th>
                <th>Match</th>
                <th>Prediction</th>
                <th>Non-Match</th>
                <th>New Prediction</th>
            </tr>
        """
        for _, row in data_df.iterrows():
            html_page += f"""
                <tr class='tr1'>
                    <td class='entity1'>
                        Entity 1
                    </td>
                    <td class='left_entity1'>
                        {row['left_description']}
                    </td>
                    <td class='left_pred' rowspan=2>
            """
            if pred_percentage:
                html_page += f"{row['start_pred']:.2%}"
            else:
                html_page += f"{row['start_pred']:.4}"

            html_page += f"""
                    </td>
                    <td class='right_entity1'>
                        {generate_strikethrough_description(row['encoded_descs'][0], row['tokens_removed'][0]) 
                        if row['encoded_descs'][0] else row['left_description']} 
                    </td>
                    <td class='right_pred' rowspan=2>
            """
            if pred_percentage:
                html_page += f"{row['new_pred']:.2%}"
            else:
                html_page += f"{row['new_pred']:.4}"

            html_page += f"""
                    </td>
                </tr>
                <tr class='tr2'>
                    <td class='entity2'>
                        Entity 2
                    </td>
                    <td class='left_entity2'>
                        {row['right_description']}
                    </td>
                    <td class='right_entity2'>
                        {generate_strikethrough_description(row['encoded_descs'][1], row['tokens_removed'][1])
                        if row['encoded_descs'][1] else row['right_description']}
                    </td>
                </tr>
            """

        html_page += '</body>'
        html_page += '</html>'

        return html_page