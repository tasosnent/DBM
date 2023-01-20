import os
import pandas as pd

# Concatenate the classification reports from C2F models for different years into a single CSV file.

suffix = 'ue' # A suffix to indicate the experiment. e.g. 'ue' stands for enhanced supervision
root_folder = "\home\RetroData"
years = {
    2007: "Dataset_SI_2007_2022...",
    2008: "Dataset_SI_2008_2022...",
    2009: "Dataset_SI_2009_2022...",
    # ...
}

report_aggreagated_df = None

for year in years:
    folder = years[year]
    report_folder = root_folder + os.path.sep + folder + os.path.sep + "fgsi" + str(year) + suffix
    report_file = report_folder + os.path.sep + "test_report.csv"
    report_df = pd.read_csv(report_file)

    if report_aggreagated_df is None:
        report_aggreagated_df = report_df
    else:
        report_aggreagated_df = pd.concat([report_aggreagated_df, report_df], axis=0)

report_aggreagated_df.to_csv(root_folder + os.path.sep + "c2f_aggregated_report_" + suffix + "_.csv",index=False)