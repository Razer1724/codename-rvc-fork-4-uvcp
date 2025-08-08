import torch
from pathlib import Path
import faiss
import gradio as gr
import traceback
import gzip


# --- Gardio stuff start ---

def run_create_uvcp_script(pth_file, index_file, output_path):
    if not pth_file:
        return "Error: A .pth file is required."

    try:
        # Get the temporary file paths from the Gradio File components
        pth_path = pth_file.name
        # The index file is optional, so it might be None
        index_path = index_file.name if index_file else None

        # The output path is also optional, an empty string should be treated as None
        output_path = output_path if output_path else None

        return create_uvcp(pth_path, index_path, output_path)
    except Exception as e:
        return f"An unexpected error occurred: {e}\n{traceback.format_exc()}"


def uvcp_tab():
    with gr.Column():
        gr.Markdown(
            """
            # UVCP Creator
            ### Unified Voice Cloning Package
            Upload a `.pth` file and optionaly an `.index` file to create a `.uvcp` file.
            """
        )
        pth_input = gr.File(
            label="Upload PTH File",
            file_types=[".pth"],
        )
        index_input = gr.File(
            label="Upload Index File (Optional)",
            file_types=[".index"],
        )
        output_path_input = gr.Textbox(
            label="Output File Path (Optional)",
            info="If left blank, the .uvcp file will be saved in the main 'logs' directory.",
            placeholder="e.g., C:/logs/my_model.uvcp",
            interactive=True,
        )
        uvcp_output_info = gr.Textbox(
            label="Output Information",
            info="The result of the operation will be displayed here.",
            value="",
            max_lines=8,
            interactive=False,
        )
        uvcp_create_button = gr.Button("Create UVCP File")
        
        uvcp_create_button.click(
            fn=run_create_uvcp_script,
            inputs=[pth_input, index_input, output_path_input],
            outputs=[uvcp_output_info],
        )

# --- Gardio stuff end ---

# --- UVCP creation start ---

def create_uvcp(pth_path, index_path=None, output_path=None):
    """Create a gzip-compressed .uvcp file that can be loaded directly by torch.load."""
    try:
        if not Path(pth_path).exists():
            return f"Error: PTH file not found: {pth_path}"

        # Load the model weights from the .pth file.
        pth_data = torch.load(pth_path, map_location="cpu", weights_only=True)
        uvcp_data = {"model_state": pth_data}

        # If an index path is provided, read and serialize the FAISS index.
        if index_path:
            if not Path(index_path).exists():
                return f"Error: Index file not found: {index_path}"
            
            index = faiss.read_index(index_path)
            uvcp_data["index_data"] = faiss.serialize_index(index)

        # Determine the final output path for the .uvcp file.
        if output_path:
            final_output_path = Path(output_path)
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            logs_dir = project_root / "logs"
            
            # Ensure the logs directory exists.
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create the new filename based on the input .pth file.
            base_name = Path(pth_path).stem
            uvcp_filename = f"{base_name}.uvcp"
            final_output_path = logs_dir / uvcp_filename
            
        # Open a gzip file stream and save the data.
        # torch.save will write compressed data to this stream. [1]
        # torch.load can automatically detect and decompress this format.
        with gzip.open(final_output_path, 'wb') as f:
            torch.save(uvcp_data, f)
        
        return f"Successfully created UVCP file: {final_output_path}"
    except Exception as e:
        return f"An error occurred during UVCP creation: {e}\n{traceback.format_exc()}"
    
if __name__ == "__main__":

    with gr.Blocks() as demo:
        uvcp_tab()

# --- UVCP creation end ---