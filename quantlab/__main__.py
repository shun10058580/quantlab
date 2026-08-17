"""支持 ``python -m quantlab`` 直接运行。"""

from quantlab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
