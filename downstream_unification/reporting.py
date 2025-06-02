import io
from io import BytesIO

import matplotlib.pyplot as plt
import openpyxl
import pandas as pd
from evaluators.multilabel_classifier_evaluation import (
    MultiLabelEval,
    print_dict_structure,
)
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as openpyxl_get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from PIL import Image as PILImage
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb


def create_overview_tab(new_wandb_run, previous_wandb_runs):
    wb = Workbook()
    ws = wb.active
    ws.title = "Overview"

    # Set Title
    ws["A1"] = "Runs for Comparison"
    ws["A1"].font = Font(size=16, bold=True)

    # Merge cells for title
    ws.merge_cells("A1:D1")
    ws["A1"].alignment = Alignment(horizontal="center")

    # Set column headers
    headers = ["Name", "Description", "Start Time", "WandB URL"]
    for col, header in enumerate(headers, start=1):
        col_letter = chr(64 + col)  # Convert column number to letter (A, B, C, D)
        ws[f"{col_letter}2"] = header
        ws[f"{col_letter}2"].font = Font(size=14, bold=True)
        ws[f"{col_letter}2"].border = Border(
            left=Side(border_style="thin"),
            right=Side(border_style="thin"),
            top=Side(border_style="thin"),
            bottom=Side(border_style="thin"),
        )
        ws[f"{col_letter}2"].alignment = Alignment(horizontal="center")

    # Define borders
    border = Border(
        left=Side(border_style="thin"),
        right=Side(border_style="thin"),
        top=Side(border_style="thin"),
        bottom=Side(border_style="thin"),
    )

    # Fill in the data starting from row 3
    all_runs = previous_wandb_runs + [new_wandb_run]
    for i, run in enumerate(all_runs):
        row = 3 + i

        # Add run data
        ws[f"A{row}"] = run.get("name")
        ws[f"B{row}"] = run.get("desc")

        # Add start time from WandB using the correct field name
        api = wandb.Api()
        ws[f"C{row}"] = api.run(run.get("run_id")).metadata.get("startedAt", "")

        # Add WandB URL
        ws[f"D{row}"] = "https://wandb.ai/" + run.get("run_id")

        # Apply border and formatting to all cells in the row
        for col in ["A", "B", "C", "D"]:
            cell = ws[f"{col}{row}"]
            cell.border = border

            # Set alignment based on column type
            if col == "B":  # Description column
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            elif col == "C":  # Start Time column
                cell.alignment = Alignment(horizontal="center")

    # Adjust column widths based on content
    for col in ["A", "B", "C", "D"]:
        max_length = max(
            len(str(ws[f"{col}{row}"].value or ""))
            for row in range(2, len(all_runs) + 3)
        )
        adjusted_width = min(max_length + 5, 50)  # Cap the maximum width
        ws.column_dimensions[col].width = adjusted_width

    # Set a fixed larger width for description column
    ws.column_dimensions["B"].width = 50

    # Return the workbook object
    return wb


def compare_multiple_dicts(dicts, ignore_keys=None):
    all_keys = set()

    # Collect all keys from all dictionaries
    for d in dicts.values():
        all_keys.update(get_all_keys(d))

    if ignore_keys:
        all_keys = [i for i in all_keys if i.split("/")[0] not in ignore_keys]

    # Compare values for each key across all dicts and filter out constant keys
    data = []
    for key in sorted(all_keys):
        row = {"Key": key}
        values = {
            run_name: convert_value(get_nested_value(d, key.split("/")))
            for run_name, d in dicts.items()
        }

        # Only add the key to the table if its values are different
        if len(set(values.values())) > 1:
            for run_name, value in values.items():
                row[run_name] = value
            data.append(row)

    # Create and return a DataFrame
    df = pd.DataFrame(data)
    return df


def get_all_keys(d, parent_key=""):
    keys = set()
    for k, v in d.items():
        new_key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):
            keys.update(get_all_keys(v, new_key))
        else:
            keys.add(new_key)
    return keys


def get_nested_value(d, keys):
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return None
    return d


def convert_value(value):
    """Convert unhashable types like lists into a hashable format for comparison."""
    if isinstance(value, list):
        return str(sorted(value))  # Convert lists to tuples (hashable)
    elif isinstance(value, dict):
        return str(value)  # Convert dicts to strings (hashable)
    else:
        return value  # Leave other types as is


def add_diff_hyperparameter(wb, new_wandb_run, previous_wandb_runs):
    api = wandb.Api()
    dicts = {
        run_info.get("name"): api.run(run_info.get("run_id")).config
        for run_info in previous_wandb_runs + [new_wandb_run]
    }
    df = compare_multiple_dicts(dicts, ["id2label", "label2id"])

    ws = wb.create_sheet("diff_hyperparameters")

    # Set Title
    ws["A1"] = "Difference in Hyperparameters"
    ws["A1"].font = Font(size=16, bold=True)

    # Write the DataFrame to the sheet, starting from row 3
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 3):
        for c_idx, value in enumerate(row, 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=str(value))

            # Apply text wrapping to all cells
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    # Add Borders to all cells with data
    border_style = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )

    for row in ws.iter_rows(
        min_row=3,
        max_row=len(df) + 3,
        min_col=1,
        max_col=len(df.columns),
    ):
        for cell in row:
            cell.border = border_style

    # Column width setting based on 70th percentile of content length
    for col in range(1, len(df.columns) + 1):
        column_letter = ws.cell(row=1, column=col).column_letter
        content_lengths = []

        # Check header length
        header_value = ws.cell(row=3, column=col).value
        if header_value:
            content_lengths.append(len(str(header_value)))

        # Collect all content lengths in this column
        for row in range(4, len(df) + 4):
            cell_value = ws.cell(row=row, column=col).value
            if cell_value:
                # For each cell, record its content length
                content_length = len(str(cell_value))

                # For very long content, consider how it would look wrapped
                if content_length > 100:
                    # For very long strings, find length of longest word
                    words = str(cell_value).split()
                    longest_word = max(len(word) for word in words) if words else 0
                    # Use max of longest word or reasonable default
                    content_lengths.append(max(longest_word, 30))
                else:
                    content_lengths.append(min(content_length, 50))

        # Determine width based on 70th percentile if we have enough data points
        if len(content_lengths) > 3:
            # Sort the lengths and get approximately the 70th percentile
            content_lengths.sort()
            percentile_idx = int(len(content_lengths) * 0.95)
            width = content_lengths[percentile_idx]
        else:
            # If not enough data points, use the median or max (with limits)
            width = max(content_lengths) if content_lengths else 20
            width = min(width, 35)  # Cap at 35 for small samples

        # Add padding and set column width
        adjusted_width = min(width + 4, 40)  # Add padding but cap at 40

        # Special case for first column (parameter names)
        if col == 1:
            adjusted_width = min(
                width + 5,
                45,
            )  # Parameter names get slightly more space

        ws.column_dimensions[column_letter].width = adjusted_width

    # Optionally, apply bold styling to the header row and center alignment
    for cell in ws[3]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

    # Set row height for better readability with wrapped text
    for row in range(4, len(df) + 4):
        # Increase row height for better readability with wrapped text
        ws.row_dimensions[row].height = 35

    return wb


def get_loss_history_plot(wandb_runs):
    api = wandb.Api()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    for run_info in wandb_runs:
        run = api.run(run_info.get("run_id"))
        run_name = run_info.get("name")  # Get run name

        # Get loss history
        train_loss_df = run.history()[["train/epoch", "train/loss"]].dropna()
        eval_loss_df = run.history()[["train/epoch", "eval/loss"]].dropna()

        # Plot training loss
        axes[0].plot(
            train_loss_df["train/epoch"],
            train_loss_df["train/loss"],
            marker="o",
            linestyle="-",
            label=run_name,
        )

        # Plot evaluation loss
        axes[1].plot(
            eval_loss_df["train/epoch"],
            eval_loss_df["eval/loss"],
            marker="o",
            linestyle="-",
            label=run_name,
        )

    # Customize subplots
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Evaluation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    return fig  # Return figure for further use


def add_loss_comparision(wb, new_wandb_run, previous_wandb_runs):
    ws = wb.create_sheet("loss_comparison")
    # Set Title
    ws["A1"] = "Loss Comparison"
    ws["A1"].font = Font(size=16, bold=True)

    # Get the loss history figure
    fig = get_loss_history_plot(previous_wandb_runs + [new_wandb_run])

    # Save figure to a BytesIO buffer
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    img_buf.seek(0)

    # Insert the image into the sheet
    img = Image(img_buf)
    ws.add_image(img, "A3")

    return wb  # Return the modified workbook


def get_model_from_wandb_inplace(wandb_runs):
    api = wandb.Api()
    for run_info in wandb_runs:
        run = api.run(run_info.get("run_id"))
        run_name = run_info.get("name")
        tokenizer_path = (
            run_info.get("tokenizer_path")
            if run_info.get("tokenizer_path")
            else run.config.get("_name_or_path")
        )

        # Find the final model artifact
        model_artifact = None
        for artifact in run.logged_artifacts():
            if artifact.type == "model" and artifact.metadata.get("final_model"):
                model_artifact = artifact
                break
        if model_artifact is None:
            raise ValueError(f"No final model artifact found in {run_name} run!")

        # Download the model artifact
        artifact_dir = model_artifact.download()

        # Load the model and tokenizer
        run_info["model"] = AutoModelForSequenceClassification.from_pretrained(
            artifact_dir,
        )
        run_info["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_path)


def add_classifcation_report_to_excel(worksheet, results, start_row, start_col):
    reports = {}

    # transfom the result report into friendy format
    for model_name, result in results.items():
        for datasplit, report_split in result.items():
            if datasplit not in reports:
                reports[datasplit] = {}
            if model_name not in reports[datasplit]:
                reports[datasplit][model_name] = {}

            reports[datasplit][model_name] = report_split["classification_report"]

    # Now create DataFrames
    dataframes = {}

    for datasplit, models in reports.items():
        model_dfs = []
        for model_name, df in models.items():
            # Create MultiIndex columns (Model Name, Metric)
            df.columns = pd.MultiIndex.from_product([[model_name], df.columns])
            model_dfs.append(df)

        # Concatenate all models side by side
        combined_df = pd.concat(model_dfs, axis=1)
        dataframes[datasplit] = combined_df

    # now add these dataframes to the excel sheet
    dataframes = dict(sorted(dataframes.items()))
    current_row = start_row
    for datasplit, df in dataframes.items():
        current_row = write_dataframe_to_excel(
            worksheet=worksheet,
            dataframe=df,
            start_row=current_row,
            start_col=start_col,
            title=f"Classification Report for {datasplit}",
        )


def write_dataframe_to_excel(
    worksheet,
    dataframe,
    start_row=1,
    start_col=1,
    title=None,
):
    """
    Write a pandas DataFrame to an Excel worksheet with proper formatting.

    Parameters:
    -----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        The worksheet to write to
    dataframe : pandas.DataFrame
        The DataFrame to write
    start_row : int
        The row to start writing at (1-indexed)
    start_col : int
        The column to start writing at (1-indexed)
    title : str, optional
        Title to add above the DataFrame

    Returns:
    --------
    int
        The next row after the DataFrame (for adding more content)
    """

    current_row = start_row

    # Add title if provided
    if title:
        cell = worksheet.cell(row=current_row, column=start_col, value=title)
        cell.font = Font(size=16, bold=True)
        current_row += 2

    # Define styles
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )
    header_fill = PatternFill(
        start_color="E0E0E0",
        end_color="E0E0E0",
        fill_type="solid",
    )
    center_alignment = Alignment(horizontal="center", vertical="center")
    bold_font = Font(bold=True)

    # Determine how many index levels and column levels
    index_levels = (
        dataframe.index.nlevels if isinstance(dataframe.index, pd.MultiIndex) else 1
    )
    column_levels = (
        dataframe.columns.nlevels if isinstance(dataframe.columns, pd.MultiIndex) else 1
    )

    # Convert to rows with index and header
    rows = list(dataframe_to_rows(dataframe, index=True, header=True))

    # Write rows to Excel
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = worksheet.cell(
                row=current_row + i,
                column=start_col + j,
                value=value,
            )
            cell.border = thin_border
            cell.alignment = center_alignment

            if i < column_levels or j < index_levels:
                cell.font = bold_font
                cell.fill = header_fill

    # Merge cells for multi-index rows (optional but improves readability)
    if isinstance(dataframe.columns, pd.MultiIndex):
        for col_level in range(column_levels):
            row_idx = current_row + col_level
            last_val = None
            span_start = None
            for col_idx in range(index_levels, len(rows[0])):
                val = worksheet.cell(row=row_idx, column=start_col + col_idx).value
                if val != last_val:
                    if span_start is not None and col_idx - span_start > 1:
                        worksheet.merge_cells(
                            start_row=row_idx,
                            start_column=start_col + span_start,
                            end_row=row_idx,
                            end_column=start_col + col_idx - 1,
                        )
                    span_start = col_idx
                    last_val = val
            if span_start is not None and col_idx - span_start >= 1:
                worksheet.merge_cells(
                    start_row=row_idx,
                    start_column=start_col + span_start,
                    end_row=row_idx,
                    end_column=start_col + col_idx,
                )

    if isinstance(dataframe.index, pd.MultiIndex):
        for row_idx in range(column_levels, len(rows)):
            for idx_level in range(index_levels):
                val = worksheet.cell(
                    row=current_row + row_idx,
                    column=start_col + idx_level,
                ).value
                span_start = row_idx
                while (
                    row_idx + 1 < len(rows)
                    and worksheet.cell(
                        row=current_row + row_idx + 1,
                        column=start_col + idx_level,
                    ).value
                    == val
                ):
                    row_idx += 1
                if row_idx > span_start:
                    worksheet.merge_cells(
                        start_row=current_row + span_start,
                        start_column=start_col + idx_level,
                        end_row=current_row + row_idx,
                        end_column=start_col + idx_level,
                    )

    # Adjust column widths
    for j in range(len(rows[0])):
        col_letter = get_column_letter(start_col + j)
        worksheet.column_dimensions[col_letter].width = 15

    return current_row + len(rows) + 2  # extra spacing


def add_metrics_comparison(
    wb,
    new_wandb_run,
    previous_wandb_runs,
    train_df,
    test_df,
    text_col,
    label_cols,
    top_ks,
    thresholds,
    key,
    plot_width=600,
    plot_height=400,
):
    wandb_runs = previous_wandb_runs + [new_wandb_run]
    get_model_from_wandb_inplace(wandb_runs)
    ws1 = wb.create_sheet("metric_comp_plots")
    ws2 = wb.create_sheet("split_based_comp_plots")
    ws3 = wb.create_sheet("metrics_df_data")
    ws4 = wb.create_sheet("classification_report")
    # Set Title
    ws1["A1"] = "Metrics Comparison Plot"
    ws1["A1"].font = Font(size=16, bold=True)
    ws2["A1"] = "Split Based Metrics Comparison Plot"
    ws2["A1"].font = Font(size=16, bold=True)
    ws4["A1"] = "Classification Report"
    ws4["A1"].font = Font(size=16, bold=True)

    results = {}
    # Get the metrics for all runs
    for run_info in wandb_runs:
        eval = MultiLabelEval(
            model=run_info["model"],
            tokenizer=run_info["tokenizer"],
            train_data_df=train_df,
            test_data_df=test_df,
            text_col=text_col,
            label_cols=label_cols,
            model_name=run_info["name"],
            thresholds=thresholds,
            top_ks=top_ks,
        )
        results[run_info["name"]] = eval.evaluate_in_memory()

    plots = MultiLabelEval.plot_comparision(
        results=results,
        output_dir=None,
        project_name=key,
        ledgend_n_cols=3,
    )

    _end_row = 1
    for basis, dx in plots["metrics_data"].items():
        _end_row = write_dataframe_to_excel(
            worksheet=ws3,
            dataframe=dx,
            start_row=_end_row,
            start_col=1,
            title=f"Metrics Comparison Data for {basis}",
        )

    # Add plots recursively to the worksheets
    add_plots_to_worksheet(
        ws1,
        plots["metric_comp_plots"],
        start_row=2,
        start_col=1,
        path=[],
        plot_width=plot_width,
        plot_height=plot_height,
    )
    add_plots_to_worksheet(
        ws2,
        plots["split_based_comparision"],
        start_row=2,
        start_col=1,
        path=[],
        plot_width=plot_width,
        plot_height=plot_height,
    )

    add_classifcation_report_to_excel(
        worksheet=ws4,
        results=results,
        start_row=2,
        start_col=1,
    )

    return wb


def add_plots_to_worksheet(
    worksheet,
    plot_dict,
    start_row,
    start_col,
    path,
    plot_width=600,
    plot_height=400,
):
    """
    Recursively add plots to the worksheet with breadcrumb path headings.

    Args:
        worksheet: Excel worksheet to add plots to
        plot_dict: Dictionary containing plots or nested plot dictionaries
        start_row: Starting row for adding content
        start_col: Starting column for adding content
        path: List tracking the current path in the plot hierarchy
        plot_width: Width of the plot in pixels (default: 600)
        plot_height: Height of the plot in pixels (default: 400)

    Returns:
        The next row to use for content
    """
    current_row = start_row

    for key, value in plot_dict.items():
        current_path = path + [key]
        breadcrumb = " > ".join(str(item) for item in current_path)

        if isinstance(value, dict):
            # If this is a nested dictionary, recursively process it
            current_row = add_plots_to_worksheet(
                worksheet,
                value,
                current_row,
                start_col + 1,
                current_path,
                plot_width,
                plot_height,
            )
        else:
            # Add breadcrumb heading
            heading_cell = worksheet.cell(row=current_row, column=start_col)
            heading_cell.value = breadcrumb
            heading_cell.font = Font(bold=True)
            current_row += 1
            # This is a plot figure, add it to the worksheet
            try:
                img = BytesIO()
                dpi = 96  # Standard screen DPI
                value.savefig(img, format="png", bbox_inches="tight", dpi=dpi)
                img.seek(0)

                # Get original size using PIL
                with PILImage.open(img) as pil_img:
                    orig_width, orig_height = pil_img.size

                aspect_ratio = orig_width / orig_height
                img.seek(0)
                # Add the plot image to the worksheet
                img_obj = Image(img)
                img_obj.width = int(plot_height * aspect_ratio)
                img_obj.height = plot_height

                worksheet.add_image(
                    img_obj,
                    f"{get_column_letter(start_col)}{current_row}",
                )

                # Calculate rows needed based on plot height and row height
                # Default Excel row height is ~15 points
                default_row_height = 15
                rows_needed = calculate_rows_for_image(plot_height, default_row_height)

                # Move down enough rows to accommodate the image
                current_row += rows_needed
            except Exception as e:
                error_cell = worksheet.cell(row=current_row, column=start_col)
                error_cell.value = f"Error adding plot: {str(e)}"
                current_row += 2

        # Add some space between plot sections
        current_row += 2

    return current_row


def calculate_rows_for_image(image_height_pixels, row_height_points, padding=1.2):
    """
    Calculate how many Excel rows are needed to display an image of given height.

    Args:
        image_height_pixels: Height of the image in pixels
        row_height_points: Height of an Excel row in points
        padding: Padding factor to ensure image fits (default: 1.2)

    Returns:
        Number of rows needed
    """
    # Convert pixels to points (1 point = 1/72 inch, and assuming ~96 pixels per inch)
    image_height_points = image_height_pixels * (72 / 96)

    # Calculate rows needed with padding
    rows_needed = int((image_height_points / row_height_points) * padding)

    # Ensure at least 1 row
    return max(1, rows_needed)


def get_column_letter(col_idx):
    """
    Convert column index to column letter (e.g., 1 -> 'A', 27 -> 'AA')
    """
    result = ""
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        result = chr(65 + remainder) + result
    return result
