import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns


class PlotExplanation:
    @staticmethod
    def plot_impacts(data, target_col, ax, title, bar_label_fontsize: int=8):
        n = len(data)
        ax.set_xlim(-0.5, 0.5)  # set x axis limits
        ax.set_ylim(-1, n)  # set y axis limits
        ax.set_yticks(range(n))  # add 0-n ticks
        ax.set_yticklabels(data[['column', 'word']].astype(str).apply(lambda x: ', '.join(x), 1))  # add y tick labels

        # define arrows
        arrow_starts = np.repeat(0, n)
        # arrow_lengths = data[target_col].values
        arrow_lengths = data[target_col].values / 2
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
                         zorder=50)  # arrow edge color
                width = arrow_lengths[i]
                offset = 40 if width > 0 else -40
                # offset = width / 2

                if width > 0.2:
                    ha = 'center'
                    width -= 0.2
                elif 0 < width < 0.2:
                    ha = 'right'
                else:
                    ha = 'left'

                color = '#ffffff' if width > 0.2 else '#000000'

                ax.annotate(format(width, '.3f'),
                            (width, i),
                            # ha='right' if width > 0 else 'left',
                            ha=ha,
                            va='center',
                            xytext=(offset, -1),
                            textcoords='offset points',
                            zorder=100,
                            color=color,
                            fontsize=bar_label_fontsize)

                # ax.text(width, i,
                #         format(width, '.3f'),
                #         # (width, i),
                #         ha='right' if width > 0 else 'left',
                #         va='center',
                #         # textcoords='offset points',
                #         zorder=100)

        # format plot
        ax.set_title(title)  # add title
        ax.axvline(x=0, color='0.9', ls='--', lw=2, zorder=0)  # add line at x=0
        ax.grid(axis='y', color='0.9', zorder=0)  # add a light grid
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
                                         'Original Tokens', 8)
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[1],
                                         'Augmented Tokens', 8)
        else:
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[0],
                                         'Original Tokens', 8)
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('l')], target_col, axes[1],
                                         'Augmented Tokens', 8)
            # fig.suptitle('Right Landmark Explanation')
        fig.tight_layout()

    @staticmethod
    def plot(exp, figsize=(16, 6), title: bool=True, y_label: str='Landmark', y_label_fontsize: int=8,
             bar_label_fontsize: int=8):
        data = exp.copy()

        if figsize[1] < len(data) / 3:
            figsize = (16, len(data) / 3)

        original_fontsize = plt.rcParams.get('font.size')
        plt.rcParams.update({'font.size': y_label_fontsize})

        data['column'] = data['column'].str.replace('left_','l_').str.replace('right_','r_')
        # initialize a plot
        if data[data['column'].str.startswith('r')]['score_right_landmark'].abs().max() > 0.01:
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize)  # create figure
            for target_col, land_side, ax in zip(['score_right_landmark', 'score_left_landmark'], ['Right', 'Left'], axes[[1,3]]):
                # sort individuals by amount of change, from largest to smallest
                side_char = land_side[0].lower()
                data = data.sort_values(by=target_col, ascending=True).reset_index(drop=True)
                PlotExplanation.plot_impacts(data[data['column'].str.startswith(side_char)], target_col, ax,
                                             'Augmented Tokens' if title else '', bar_label_fontsize=bar_label_fontsize)
            axes_for_original = axes[[0,2]]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)  # create figure
            axes_for_original = axes

        for target_col, land_side, ax in zip(['score_right_landmark', 'score_left_landmark'],['Right', 'Left'], axes_for_original):
            # sort individuals by amount of change, from largest to smallest
            side_char = land_side[0].lower()
            opposite_side_char = 'r' if side_char == 'l' else 'l'
            data = data.sort_values(by=target_col, ascending=True).reset_index(drop=True)
            PlotExplanation.plot_impacts(data[data['column'].str.startswith(opposite_side_char)], target_col, ax,
                                         'Original Tokens' if title else '', bar_label_fontsize=bar_label_fontsize)
            # if data[data['column'].str.startswith(side_char)][target_col].abs().max()>0.05:
            #     PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[1], 'Augmented Tokens')
            ax.set_ylabel(f'{land_side} {y_label}')
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

        plt.rcParams.update({'font.size': original_fontsize})

        return fig, axes

    @staticmethod
    def plot_counterfactual(data_df: pandas.DataFrame, pred_percentage: bool=True,
                            palette: list=sns.color_palette().as_hex(), positive_examples: bool=True,
                            spaced_attributes: bool=True, beautify_table: bool=True, align_left: bool=False) -> str:

        whitespace = ' '

        def generate_strikethrough_description(encoded_description, tokens_to_remove):
            new_description = str()

            for desc_token in encoded_description:
                if desc_token == '<br>':
                    new_description += desc_token
                else:
                    token = desc_token.split('">')[-1]  # remove first font tag part, but still has </font> suffix
                    token = token.split('<')[0] # remove suffix
                    cleaned_token = token.split('_')[-1]  # remove wym prefix
                    first_part_color_tag, second_part_color_tag = desc_token.split(token)  # recompute the html font tag

                    if token in tokens_to_remove:
                        new_description = whitespace.join([new_description, f'<del>{cleaned_token}</del>'])
                    else:
                        cleaned_token = first_part_color_tag + cleaned_token + second_part_color_tag
                        new_description = whitespace.join([new_description, cleaned_token])

            return new_description

        def color_descriptions(df_to_color, left_colors, right_colors):
            left_starting_prefix = 'A'
            right_starting_prefix = 'A'

            descs_df = pandas.DataFrame(df_to_color['encoded_descs'].to_list(),
                                    columns=['left_encoded_descs', 'right_encoded_descs'])

            if descs_df['left_encoded_descs'].apply(len).gt(0).any():
                right_starting_prefix = chr(ord(right_starting_prefix) + len(left_colors))

            tmp_df = df_to_color.copy()

            left_attr_to_prefix = {attribute: chr(ord(left_starting_prefix) + prefix_idx)
                                   for prefix_idx, (attribute, _) in enumerate(left_colors)}


            right_attr_to_prefix = {attribute: chr(ord(right_starting_prefix) + prefix_idx)
                                   for prefix_idx, (attribute, _) in enumerate(right_colors)}

            for index, row in tmp_df.iterrows():
                new_left_description = list()
                new_left_encoded_description = list()
                new_right_description = list()
                new_right_encoded_description = list()
                left_encoded_description, right_encoded_description = row['encoded_descs']

                for attribute, color in left_colors:
                    new_left_description.append(f'<font color="{color}">{row[attribute]}</font>')

                    if left_encoded_description:
                        attribute_prefix = left_attr_to_prefix[attribute]

                        for encoded_token in left_encoded_description:
                            if encoded_token.startswith(attribute_prefix):
                                new_left_encoded_description.append(f'<font color="{color}">{encoded_token}</font>')

                        if left_encoded_description:
                            if spaced_attributes:
                                new_left_encoded_description.append('<br>')

                    if spaced_attributes:
                        new_left_description.append('<br>')

                for attribute, color in right_colors:
                    new_right_description.append(f'<font color="{color}">{row[attribute]}</font>')

                    if right_encoded_description:
                        attribute_prefix = right_attr_to_prefix[attribute]

                        for encoded_token in right_encoded_description:
                            if encoded_token.startswith(attribute_prefix):
                                new_right_encoded_description.append(f'<font color="{color}">{encoded_token}</font>')

                        if right_encoded_description:
                            if spaced_attributes:
                                new_right_encoded_description.append('<br>')

                    if spaced_attributes:
                        new_right_description.append('<br>')

                df_to_color.at[index, 'left_description'] = whitespace.join(new_left_description)
                df_to_color.at[index, 'right_description'] = whitespace.join(new_right_description)
                df_to_color.at[index, 'encoded_descs'] = [new_left_encoded_description, new_right_encoded_description]

            return df_to_color


        left_columns = [column for column in data_df.columns if column.startswith('left_')
                        and column not in ('left_id', 'left_description')]
        right_columns = [column for column in data_df.columns if column.startswith('right_')
                         and column not in ('right_id', 'right_description')]
        left_number_of_colors = len(left_columns)
        right_number_of_colors = len(right_columns)
        palette_length = len(palette)

        if palette_length < left_number_of_colors:
            raise ValueError(f"Palette length ({palette_length}) is smaller than the number of "
                             f"left attributes ({left_number_of_colors}).")

        if palette_length < right_number_of_colors:
            raise ValueError(f"Palette length ({palette_length}) is smaller than the number of "
                             f"right attributes ({right_number_of_colors}).")

        if palette_length > left_number_of_colors:
            left_palette = palette[:left_number_of_colors]
        else:
            left_palette = palette

        if palette_length > right_number_of_colors:
            right_palette = palette[:right_number_of_colors]
        else:
            right_palette = palette

        left_colors = [(attribute, color) for attribute, color in zip(left_columns, left_palette)]
        right_colors = [(attribute, color) for attribute, color in zip(right_columns, right_palette)]

        df_to_color = data_df.copy()

        colored_df = color_descriptions(df_to_color, left_colors, right_colors)

        html_page = '<html>'
        html_page += """
        <head>
            <style>
            
                #first_row {
                    border-radius: 15px 15px 0px 0px;
                    border: 2px solid black;
                }
                
                #first_row th {
                    border: 2px solid black;
                }
                
                #first_row th:first-child{
                    border-radius: 15px 0 0 0;
                }
                
                #first_row th:last-child{
                    border-radius: 0 15px 0 0;
                }
                                
                table, tr, td, th {
                    text-align: center;
                    border-collapse: collapse;
                }
                
                tr, td, th{
                    border: 1px solid gray;
                }
                
                table{
                    border: 3px solid black;
                    border-radius: 15px 15px 15px 15px;
                }
                
                #container{
                    margin: 0 auto;
                    /* min-width: 100%; */
                }
                
                table tbody tr:last-child{
                    border-bottom: 2px solid black;
                }
                
                del {
                    color: red;
                }
                
                .tr2 {
                    border-bottom: 2px solid black;
                }
                
                .tr1 {
                    border-top: 2px solid black;
                }
                
                .pred {
                    border: 2px solid black;
                }
                
                .entity1, .entity2 {
                    border-left: 2px solid black;
                    border-right: 2px solid black;
                }
                
                html{
                    font-family: Arial, sans-serif;
                }
                
            </style>
        </head>
        """
        html_page += '<body>'
        html_page += '<span id="container">'

        table_style = 'style="{}"'
        table_css_properties = str()

        if beautify_table:
            table_css_properties += "min-width: 70%;"

        if not align_left:
            table_css_properties = whitespace.join([table_css_properties, "margin: 0 auto;"])

        if not table_css_properties:
            table_style = str()

        else:
            table_style = table_style.format(table_css_properties)

        td_style = 'style="padding: 10px;"' if beautify_table else ''
        html_page += f'<table {table_style}>'
        html_page += f"""
            <tr id="first_row">
                <th></th>
                <th>{"Match" if positive_examples else "Non-Match"}</th>
                <th>Prediction</th>
                <th>{"Non-Match" if positive_examples else "Match"}</th>
                <th>New<br>Prediction</th>
            </tr>
        """

        for _, row in colored_df.iterrows():
            left_description = row['left_description']
            right_description = row['right_description']
            start_pred = row['start_pred'] if positive_examples else row['new_pred']
            new_pred = row['new_pred'] if positive_examples else row['start_pred']
            left_encoded_desc, right_encoded_desc = row['encoded_descs']
            evaluate_removing_du = False

            # check if evaluation with decision units
            if len(row['tokens_removed']) == 2:
                left_tokens_removed, right_tokens_removed = row['tokens_removed']

                # if evaluation with decision units
                if type(left_tokens_removed) == list and type(right_tokens_removed) == list:
                    evaluate_removing_du = True

            if evaluate_removing_du:
                left_tokens_removed, right_tokens_removed = row['tokens_removed']

                if left_encoded_desc:
                    left_strikethrough_desc = generate_strikethrough_description(left_encoded_desc,
                                                                                 left_tokens_removed)
                else:
                    left_strikethrough_desc = left_description

                if right_encoded_desc:
                    right_strikethrough_desc = generate_strikethrough_description(right_encoded_desc,
                                                                                  right_tokens_removed)
                else:
                    right_strikethrough_desc = right_description

            # tokens_removed has a different length than 2, or if it contains exactly two elements they are strings,
            # thus it can't be a list of two lists. The evaluation was made with evaluate_removing_du = False
            else:
                left_tokens_removed = row['tokens_removed'] if left_encoded_desc else None
                right_tokens_removed = row['tokens_removed'] if right_encoded_desc else None

                if left_encoded_desc:
                    left_strikethrough_desc = generate_strikethrough_description(left_encoded_desc,
                                                                                 left_tokens_removed)
                else:
                    left_strikethrough_desc = left_description

                if right_encoded_desc:
                    right_strikethrough_desc = generate_strikethrough_description(right_encoded_desc,
                                                                                  right_tokens_removed)
                else:
                    right_strikethrough_desc = right_description

            html_page += f"""
                <tr class='tr1'>
                    <td class='entity1' {td_style}>
                        Entity 1
                    </td>
                    <td class='left_entity1' {td_style}>
                        {left_description}
                    </td>
                    <td class="pred" rowspan=2 {td_style}>
            """
            if pred_percentage:
                html_page += f"{start_pred:.2%}"
            else:
                html_page += f"{start_pred:.4}"

            html_page += f"""
                    </td>
                    <td class='right_entity1' {td_style}>
                        {left_strikethrough_desc} 
                    </td>
                    <td class="pred" rowspan=2 {td_style}>
            """
            if pred_percentage:
                html_page += f"{new_pred:.2%}"
            else:
                html_page += f"{new_pred:.4}"

            html_page += f"""
                    </td>
                </tr>
                <tr class='tr2'>
                    <td class='entity2' {td_style}>
                        Entity 2
                    </td>
                    <td class='left_entity2' {td_style}>
                        {right_description}
                    </td>
                    <td class='right_entity2' {td_style}>
                        {right_strikethrough_desc}
                    </td>
                </tr>
            """

        html_page += "</table>"
        html_page += "</span>"
        html_page += '</body>'
        html_page += '</html>'

        return html_page