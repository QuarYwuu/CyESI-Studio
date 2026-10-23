"""Launch Studio or dispatch command-line simulation options."""
def main():
    import sys
    from ..main import main as dispatch
    if len(sys.argv)==1:sys.argv.append('--studio')
    dispatch()
