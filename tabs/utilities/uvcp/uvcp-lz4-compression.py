import torch
from pathlib import Path
import faiss
import gradio as gr
import traceback
import lz4.frame


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
    """Create a compressed .uvcp file with a lossless algorithm."""
    try:
        if not Path(pth_path).exists():
            return f"Error: PTH file not found: {pth_path}"

        pth_data = torch.load(pth_path, map_location="cpu", weights_only=True)
        uvcp_data = {"model_state": pth_data}

        if index_path:
            if not Path(index_path).exists():
                return f"Error: Index file not found: {index_path}"
            
            index = faiss.read_index(index_path)
            uvcp_data["index_data"] = faiss.serialize_index(index)

        if output_path:
            final_output_path = Path(output_path)
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            logs_dir = project_root / "logs"
            
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(pth_path).stem
            uvcp_filename = f"{base_name}.uvcp"
            final_output_path = logs_dir / uvcp_filename
            
        # Save the data with lz4 lossless compression
        with lz4.frame.open(final_output_path, mode='wb') as f:
            torch.save(uvcp_data, f)
        
        return f"Successfully created UVCP file: {final_output_path}"
    except Exception as e:
        return f"An error occurred during UVCP creation: {e}\n{traceback.format_exc()}"

def load_uvcp_fast(uvcp_path):
    """
    Loads a compressed .uvcp file with optimizations for speed.
    """
    try:
        with lz4.frame.open(uvcp_path, mode='rb') as f:
            # Using mmap=True can speed up loading of large tensors
            # by memory-mapping them instead of loading them into RAM all at once. [1]
            uvcp_data = torch.load(f, map_location="cpu", mmap=True)

        model_state = uvcp_data.get("model_state")
        index = None

        if "index_data" in uvcp_data:
            # Deserialize the FAISS index from the byte data in the dictionary.
            index_buffer = uvcp_data["index_data"]
            reader = faiss.PyIOReader(faiss.py_buffer_to_vector(index_buffer))
            index = faiss.read_index(reader)

        return model_state, index
    except Exception as e:
        return f"An error occurred during UVCP loading: {e}\n{traceback.format_exc()}", None
    
if __name__ == "__main__":

    with gr.Blocks() as demo:
        uvcp_tab()

# --- UVCP creation end ---