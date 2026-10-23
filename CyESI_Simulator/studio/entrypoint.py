"""English Studio launcher; numerical command-line options remain available."""
def main():
    import sys
    from ..main import main as dispatch
    if len(sys.argv)==1:sys.argv.append('--studio')
    dispatch()
