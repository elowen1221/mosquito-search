"""mosquito：GitHub 仓库发现 MVP。"""

import uvicorn


def main() -> None:
    uvicorn.run("mosquito.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()