import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os

def browse_file(var):
    path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xls *.xlsx")])
    if path:
        var.set(path)

def browse_folder(var):
    path = filedialog.askdirectory()
    if path:
        var.set(path)

FILTER_OPTIONS = {
    1: ["Pos", "Neg"],
    2: ["all Pos", "Pos/Neg"],
    3: ["all Pos", "Pos/Pos/Neg" , "Pos/Neg/Pos"]
}

def update_filter_options(*args):
    num_channels = num_channels_var.get()
    menu = filter_option_menu['menu']
    menu.delete(0, 'end')
    for option in FILTER_OPTIONS[num_channels]:
        menu.add_command(label=option, command=tk._setit(filter_var, option))
    filter_var.set(FILTER_OPTIONS[num_channels][0])



def run_pipeline():
    input_path = input_var.get()
    output_dir = output_var.get()
    filter_option = filter_var.get()
    num_channels = num_channels_var.get()

    if not (input_path and output_dir):
        messagebox.showerror("Error", "Please select input file and output directory")
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
        script_dir = "C:\\Users\\97252\\OneDrive\\Desktop\\final project\\T_IL-Celomics-7-C\\filtering_by_protein_expression"

        # Helper function to process a single file
        def process_file(input_file):
            filename = os.path.basename(input_file)
            suffix_parts = filename.split('_')
            suffix = suffix_parts[-2] if len(suffix_parts) >= 2 else ""

            channel_part = "_".join([f"Cha{i}" for i in range(1, num_channels + 1)])
            categorized_file = os.path.join(
                output_dir,
                f"Segmented_Intensity_{channel_part}_{suffix}.xlsx"
            )

            if not os.path.exists(categorized_file):
                subprocess.run(["python", os.path.join(script_dir, "code_imaris_step1.py"), input_file, output_dir, suffix,
                                str(num_channels)], check=True)
            else:
                print(f"File {categorized_file} already exists, skipping step 1.")
            
            combined_filename = f"Gab_Normalized_Combined_{suffix}.xlsx"
            combined_path = os.path.join(output_dir, combined_filename)

            if not os.path.exists(combined_path):
                subprocess.run(["python", os.path.join(script_dir, "Categorized_step2.py"), input_file, categorized_file, output_dir, suffix,
                                str(num_channels)], check=True)
            else:
                print(f"File {combined_path} already exists, skipping step 2.")
            # Run the filter_data script with the appropriate arguments
            # Determine the filter argument based on the number of channels and selected filter option
            filter_arg = ""
            if num_channels == 1:
                if filter_option == "Pos":
                    filter_arg = "P"
                elif filter_option == "Neg":
                    filter_arg = "N"
            elif num_channels == 2:
                if filter_option == "Pos/Neg":
                    filter_arg = "PN"
                elif filter_option == "all Pos":
                    filter_arg = "PP"
            elif num_channels == 3:
                if filter_option == "all Pos":
                    filter_arg = "PPP"
                elif filter_option == "Pos/Pos/Neg":
                    filter_arg = "PPN"
                elif filter_option == "Pos/Neg/Pos":
                    filter_arg = "PNP"
            
            subprocess.run(["python", os.path.join(script_dir, "filter_data.py"), output_dir, suffix, filter_arg, str(num_channels)], check=True)

        # If input_path is a directory, process all Excel files inside
        if os.path.isdir(input_path):
            excel_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.xls', '.xlsx'))]
            if not excel_files:
                messagebox.showerror("Error", "No Excel files found in the selected directory.")
                return
            for file in excel_files:
                print(f"Processing file: {file}")
                process_file(file)
        else:
            print(f"Processing single file: {input_path}")
            process_file(input_path)
        messagebox.showinfo("Success", "Filtering complete!")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Script Error", f"Step failed: {e}")

root = tk.Tk()
root.title("Channel Processing GUI")

input_var = tk.StringVar()
output_var = tk.StringVar()
filter_var = tk.StringVar(value="all Pos")# Default to first option
num_channels_var = tk.IntVar(value=2)  # Default to 2 channels
num_channels_var.trace_add('write', update_filter_options)

# Input file/folser
tk.Label(root, text="Input Excel File or Folder:").grid(row=0, column=0, sticky='e')
tk.Entry(root, textvariable=input_var, width=60).grid(row=0, column=1)
tk.Button(root, text="Browse File", command=lambda: browse_file(input_var)).grid(row=0, column=2)
tk.Button(root, text="Browse Folder", command=lambda: browse_folder(input_var)).grid(row=0, column=3)

# Output folder
tk.Label(root, text="Output Folder:").grid(row=1, column=0, sticky='e')
tk.Entry(root, textvariable=output_var, width=60).grid(row=1, column=1)
tk.Button(root, text="Browse", command=lambda: browse_folder(output_var)).grid(row=1, column=2)

# Filter option (dynamic)
tk.Label(root, text="Filter Type (NIR/GREEN/RED):").grid(row=2, column=0, sticky='e')
filter_option_menu = tk.OptionMenu(root, filter_var, *FILTER_OPTIONS[num_channels_var.get()])
filter_option_menu.grid(row=2, column=1, sticky='w')


# Warning label
warning_text = ("⚠️ Please select the correct number of channels for your input file.\n"
                "If the number of channels does not match the file, the code may not work as expected. ⚠️")
tk.Label(root, text=warning_text, fg="red", justify="left", wraplength=400).grid(row=4, column=0, columnspan=3, pady=(10,0))

# Number of channels
tk.Label(root, text="Number of Channels:").grid(row=3, column=0, sticky='e')
tk.OptionMenu(root, num_channels_var, 1, 2, 3).grid(row=3, column=1, sticky='w')



# Run button
tk.Button(root, text="Run Full Process", command=run_pipeline, bg="lightgreen").grid(row=3, column=1, pady=20)

root.mainloop()
