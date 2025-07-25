# config.py

target_medicines = [
    'Aceta',
    'Fixal',
    'Disopan',
    'Telfast',
    'Sergel'
]

characters = sorted(list(set("".join(target_medicines))))
image_height = 32
image_width = 128
model_path = "models/crnn_model.pth"