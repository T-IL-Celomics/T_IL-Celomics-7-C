import pandas as pd
import sys
import time
import os


def main(output_file, imaris_files):
    startrow = 0
    if os.path.exists(output_file):
        os.remove(output_file)
    for imaris_file in imaris_files:
        print("Working on " + imaris_file)
        print("startrow =", startrow)
        summary_table_path = output_file.split(".")[0]
        if "_FULL" in summary_table_path:
        	summary_table_path = summary_table_path[:-5]
        file_path = summary_table_path+"_"+imaris_file.split(".")[0]+".xlsx"
        to_merge = pd.read_excel(file_path)
        to_merge.drop(columns="Unnamed: 0", inplace=True)
        if startrow == 0:
            with pd.ExcelWriter(output_file, mode="w", engine="openpyxl") as writer:
                to_merge.to_excel(writer)
            startrow = len(to_merge) + 1
        else:
            with pd.ExcelWriter(output_file, mode="a", engine="openpyxl") as writer:
                writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
                to_merge.to_excel(writer, sheet_name="Sheet1", header=False, startrow=startrow)
            startrow += len(to_merge)


if __name__ == "__main__":
    print("Starting to merge your files!")
    try:
        output_file = sys.argv[1]
        imaris_files = sys.argv[2:]
        if output_file[-5:] != ".xlsx":
            raise TypeError("Bad output_file path.")
        main(output_file, imaris_files)
        print("All done here!")
    except Exception:
        print("COULDN'T MERGE - BAD INPUT:\n")
        raise
    finally:
        time.sleep(5)



