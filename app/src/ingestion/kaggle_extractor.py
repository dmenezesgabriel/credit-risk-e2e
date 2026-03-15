import base64
import io
import logging
import os
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("KaggleExtractor")


class ExtractionConfig(ABC):
    pass


class DataExtractionStrategy(ABC):
    @abstractmethod
    def download_dataset(self, config: ExtractionConfig) -> None:
        pass


@dataclass(frozen=True)
class KaggleExtractionConfig(ExtractionConfig):
    dataset_slug: str  # e.g. "zynicide/wine-reviews"
    file_name: str  # file inside the Kaggle zip
    destination_path: str  # folder to extract to
    output_file_name: str | None = None  # optional rename


class KaggleDataExtractor(DataExtractionStrategy):
    def __init__(self, username: str, api_token: str) -> None:
        self.username = username
        self.api_token = api_token
        self.auth_header = self._create_auth_header()

    def _create_auth_header(self):
        token = f"{self.username}:{self.api_token}"
        base64_token = base64.b64encode(token.encode()).decode()
        return {"Authorization": f"Basic {base64_token}"}

    def download_dataset(self, config: ExtractionConfig) -> None:
        if not isinstance(config, KaggleExtractionConfig):
            raise TypeError("config must be a KaggleExtractionConfig instance")

        url = f"https://www.kaggle.com/api/v1/datasets/download/{config.dataset_slug}"
        request = Request(url, headers=self.auth_header)

        logger.info(f"Starting download from Kaggle: {url}")

        try:
            with urlopen(request) as response:
                data = response.read()
            logger.info("Download completed. Extracting zip file...")

            os.makedirs(config.destination_path, exist_ok=True)

            with zipfile.ZipFile(io.BytesIO(data)) as z:
                extracted_path = z.extract(
                    config.file_name, path=config.destination_path
                )

            if config.output_file_name is None:
                logger.info(
                    f"Dataset '{config.file_name}' extracted successfully "
                    f"to: {config.destination_path}"
                )
                return

            old_path = os.path.join(config.destination_path, config.file_name)
            new_path = os.path.join(
                config.destination_path, config.output_file_name
            )

            os.rename(old_path, new_path)

            logger.info(
                f"Dataset '{config.file_name}' extracted successfully "
                f"to: {config.destination_path}"
            )

        except HTTPError as e:
            logger.error(f"HTTP Error {e.code}: {e.reason}")
        except URLError as e:
            logger.error(f"URL Error: {e.reason}")
        except zipfile.BadZipFile:
            logger.error(
                "Failed to read zip file. Kaggle may have returned HTML instead of a zip."
            )
        except Exception as e:
            logger.exception(f"Unexpected error occurred: {e}")
