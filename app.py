from flask import Flask, request, jsonify
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from gtts import gTTS
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load chart-to-table model for extracting data from image
chart_to_table_model_name = "khhuang/chart-to-table"
chart_to_table_model = VisionEncoderDecoderModel.from_pretrained(chart_to_table_model_name)

# Initialize the processor
processor = DonutProcessor.from_pretrained(chart_to_table_model_name)

def clean_data(extracted_table):
    # Remove symbolic elements
    cleaned_data = extracted_table.replace("&", "").replace("|", "").strip()
    
    # Split into rows
    rows = cleaned_data.split("&&&")
    
    # Format rows and columns
    formatted_data = []
    for row in rows:
        columns = row.split("|")
        formatted_row = [col.strip() for col in columns if col.strip()]
        formatted_data.append(formatted_row)
    
    return formatted_data

@app.route('/api/process_image', methods=['POST'])
def process_image():
    # Get the image file from the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the image file temporarily
        image_path = "temp_image.png"
        file.save(image_path)
        
        # Format text inputs for chart-to-table model
        input_prompt = "<data_table_generation> <s_answer>"

        # Encode chart figure and tokenize text
        img = Image.open(image_path)
        pixel_values = processor(img.convert("RGB"), random_padding=False, return_tensors="pt").pixel_values
        decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt", max_length=100).input_ids

        # Generate table from image
        outputs = chart_to_table_model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=chart_to_table_model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=4,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # Decode and extract table from output
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        extracted_table = sequence.split("<s_answer>")[1].strip()

        # Clean and format the extracted table data
        formatted_data = clean_data(extracted_table)

        # Convert text to speech using gTTS
        tts = gTTS(text=extracted_table, lang='en')
        tts_path = "extracted_summary.mp3"
        tts.save(tts_path)

        # Clean up the temporary image file
        os.remove(image_path)

        return jsonify({'extracted_table': formatted_data, 'tts_path': tts_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')