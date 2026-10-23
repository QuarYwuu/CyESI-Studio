from CyESI_Simulator.main import main

if __name__ == "__main__":
    import sys
    if len(sys.argv)==1:sys.argv.append('--studio')
    main()
