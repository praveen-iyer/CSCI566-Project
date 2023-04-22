import os
import time

from utils.evaluation_utils import content_fidelity_files, global_effects_files, local_patterns_files


if __name__=="__main__":
    data_path = os.path.join(os.path.dirname(__file__), "data")
    content_path = os.path.join(data_path, "content-images", "golden_gate.jpg")
    style1_path = os.path.join(data_path, "style-images", "udnie.jpg")
    style2_path = os.path.join(data_path, "style-images", "mosaic.jpg")
    # output_path = os.path.join(data_path, "output-images", "combined_mo-net_golden_gate_udnie_mosaic", "0102.jpg")
    output_path = os.path.join(data_path, "output-images", "combined_cascade-net_golden_gate_udnie_mosaic", "0102_2.jpg")

    start = time.time()
    cf = content_fidelity_files(output_path, content_path)
    print(f"Content Fidelity={cf} calculated in {time.time() - start}s")
    
    start = time.time()
    ge_style1 = global_effects_files(output_path, style1_path)
    ge_style2 = global_effects_files(output_path, style2_path)
    print(f"Global Effects: style1={ge_style1}, style2={ge_style2} calculated in {time.time()-start}s")
    
    start = time.time()
    lp_style1 = local_patterns_files(output_path, style1_path)
    lp_style2 = local_patterns_files(output_path, style2_path)
    print(f"Local Patterns: style1={lp_style1}, style2={lp_style2} calculated in {time.time() - start}s")