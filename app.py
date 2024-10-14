import gradio as gr

from replacer import pipeline

interface = gr.Interface(
    fn=pipeline,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    flagging_mode="never"
)


if __name__ == "__main__":
    interface.launch(share=True)
