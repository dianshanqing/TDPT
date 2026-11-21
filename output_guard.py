"""Output contract for auxiliary CLI tools (old input may be read explicitly)."""
from runtime_identity import validate_output_path,assert_new_file,prepare_run_directory

def guard_output_arguments(args):
    # Validate every destination before creating any of them.
    normalized={}
    for key in ('outdir','out_json','out_csv','out'):
        value=getattr(args,key,None)
        if value:
            normalized[key]=str(validate_output_path(value) if key=='outdir' else assert_new_file(value))
    for key,value in normalized.items():setattr(args,key,value)
    if getattr(args,'outdir',None):args.outdir=str(prepare_run_directory(args.outdir))
    return args
