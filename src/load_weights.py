import json
import numpy as np

# Load weights and convert to C header file
with open('models/weights.json', 'r') as f:
    weights = json.load(f)

with open('src/weights.h', 'w') as f:
    f.write('#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n')
    
    for name, data in weights.items():
        arr = np.array(data)
        flat = arr.flatten()
        cname = name.replace('.', '_')
        f.write(f'static const float {cname}[] = {{\n')
        for i, v in enumerate(flat):
            f.write(f'  {v:.6f}f,')
            if (i+1) % 8 == 0: f.write('\n')
        f.write('\n};\n\n')
    
    f.write('#endif\n')

print("weights.h generated!")
