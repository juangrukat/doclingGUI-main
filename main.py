import sys
import os
import logging
import shutil
from pathlib import Path
import tempfile
from typing import Optional, List, Set
from dataclasses import dataclass
import subprocess
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QFileDialog, QCheckBox, QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import json
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('docling_gui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    input_path: str
    output_path: str
    export_format: str
    table_mode: str
    force_ocr: bool
    ocr_bitmaps: bool
    temp_dir: str
    accelerator: str
    ocr_languages: str
    workers: int
    # EasyOCR parameters
    text_threshold: float
    low_text: float
    link_threshold: float
    canvas_size: int
    mag_ratio: float
    # Bounding box parameters
    slope_ths: float
    ycenter_ths: float
    height_ths: float
    width_ths: float
    # Additional parameters
    contrast_ths: float
    adjust_contrast: float
    batch_size: int
    decoder: str

class FileProcessor:
    """Handles file processing operations."""
    
    VALID_EXTENSIONS: Set[str] = {
        '.pdf', '.docx', '.pptx', '.html', '.xlsx', '.md', '.txt'
    }

    @staticmethod
    def is_valid_file(file_path: Path) -> bool:
        """Check if file is valid for processing."""
        return (
            not file_path.name.startswith('.') and 
            file_path.suffix.lower() in FileProcessor.VALID_EXTENSIONS
        )

    @staticmethod
    def get_files_to_process(directory: Path) -> List[Path]:
        """Get list of valid files to process from directory."""
        files = []
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file() and FileProcessor.is_valid_file(file_path):
                    files.append(file_path)
                    logger.debug(f"Found valid file: {file_path}")
        except Exception as e:
            logger.error(f"Error scanning directory: {str(e)}")
        return files

class DoclingWorker(QThread):
    """Worker thread for processing documents."""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, config: ProcessingConfig):
        super().__init__()
        self.config = config
        self.is_cancelled = False

    def run(self) -> None:
        """Main processing loop."""
        try:
            input_dir = Path(self.config.input_path)
            files_to_process = FileProcessor.get_files_to_process(input_dir)
            
            for file_path in files_to_process:
                if self.is_cancelled:
                    logger.info("Processing cancelled by user")
                    break

                self.progress.emit(f"Processing {file_path}")
                self._build_docling_command(file_path)
                
        except Exception as e:
            logger.error(f"Error in worker thread: {str(e)}")
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def _build_docling_command(self, file_path: Path) -> List[str]:
        """Build docling command with appropriate options."""
        # Create pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        # Configure EasyOCR options with only the supported parameters
        ocr_options = EasyOcrOptions(
            force_full_page_ocr=self.config.force_ocr
        )

        # Set additional options through environment variables
        os.environ.update({
            'EASYOCR_TEXT_THRESHOLD': str(self.config.text_threshold),
            'EASYOCR_LOW_TEXT': str(self.config.low_text),
            'EASYOCR_LINK_THRESHOLD': str(self.config.link_threshold),
            'EASYOCR_CANVAS_SIZE': str(self.config.canvas_size),
            'EASYOCR_MAG_RATIO': str(self.config.mag_ratio),
            'EASYOCR_SLOPE_THS': str(self.config.slope_ths),
            'EASYOCR_YCENTER_THS': str(self.config.ycenter_ths),
            'EASYOCR_HEIGHT_THS': str(self.config.height_ths),
            'EASYOCR_WIDTH_THS': str(self.config.width_ths),
            'EASYOCR_CONTRAST_THS': str(self.config.contrast_ths),
            'EASYOCR_ADJUST_CONTRAST': str(self.config.adjust_contrast),
            'EASYOCR_BATCH_SIZE': str(self.config.batch_size),
            'EASYOCR_DECODER': self.config.decoder,
            'EASYOCR_NUM_WORKERS': str(self.config.workers)
        })

        pipeline_options.ocr_options = ocr_options

        # Create document converter with options
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

        # Convert document
        try:
            result = converter.convert(file_path)
            # Export to specified format
            if self.config.export_format == 'md':
                output = result.document.export_to_markdown()
            elif self.config.export_format == 'json':
                output = result.document.export_to_json()
            elif self.config.export_format == 'text':
                output = result.document.export_to_text()
            
            # Write output to file
            output_path = Path(self.config.output_path) / f"{file_path.stem}.{self.config.export_format}"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output)
                
            return True
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            self.error.emit(str(e))
            return False
        finally:
            # Clean up environment variables
            for key in os.environ:
                if key.startswith('EASYOCR_'):
                    del os.environ[key]

    def cancel(self) -> None:
        """Cancel processing."""
        self.is_cancelled = True

class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.temp_dir: Optional[str] = None
        self.worker: Optional[DoclingWorker] = None
        self.init_ui()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle('Docling GUI')
        self.setMinimumWidth(600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add UI components
        self._setup_folder_selection(layout)
        self._setup_processing_options(layout)
        self._setup_progress_section(layout)
        self._setup_control_buttons(layout)

    def _setup_folder_selection(self, layout: QVBoxLayout) -> None:
        """Setup input/output folder selection."""
        self.input_path = QLabel('No folder selected')
        self.output_path = QLabel('No folder selected')

        for label, path_label, button_text in [
            ('Input:', self.input_path, 'Select Input Folder'),
            ('Output:', self.output_path, 'Select Output Folder')
        ]:
            h_layout = QHBoxLayout()
            h_layout.addWidget(QLabel(label))
            h_layout.addWidget(path_label)
            btn = QPushButton(button_text)
            btn.clicked.connect(
                self.select_input_folder if 'Input' in button_text 
                else self.select_output_folder
            )
            h_layout.addWidget(btn)
            layout.addLayout(h_layout)

    def _setup_processing_options(self, layout: QVBoxLayout) -> None:
        """Setup processing options."""
        # Basic Options Section
        layout.addWidget(self._create_section_label("Basic Options"))
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(['md', 'json', 'text', 'doctags'])
        
        self.table_mode = QComboBox()
        self.table_mode.addItems(['fast', 'accurate'])
        
        basic_tooltips = {
            'Export Format': 'Output format:\n- md: Markdown with formatting\n- json: Structured data\n- text: Plain text only',
            'Table Mode': 'Table detection:\n- fast: Quick but basic\n- accurate: Better but slower'
        }
        
        for label, combo, tooltip in [
            ('Export Format:', self.format_combo, basic_tooltips['Export Format']),
            ('Table Mode:', self.table_mode, basic_tooltips['Table Mode'])
        ]:
            self._add_labeled_widget(layout, label, combo, tooltip)

        # OCR Options Section
        layout.addWidget(self._create_section_label("OCR Settings"))
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel('OCR Languages:'))
        self.language_input = QLineEdit()
        self.language_input.setPlaceholderText('e.g., en,es,fr')
        self.language_input.setText('en')
        self.language_input.setToolTip('Comma-separated language codes (e.g., en,es,fr)')
        h_layout.addWidget(self.language_input)
        layout.addLayout(h_layout)
        
        self.force_ocr = QCheckBox('Force OCR')
        self.force_ocr.setToolTip('Always use OCR, even if text can be extracted directly')
        self.ocr_bitmaps = QCheckBox('OCR for Bitmaps')
        self.ocr_bitmaps.setToolTip('Apply OCR to bitmap images in documents')
        
        layout.addWidget(self.force_ocr)
        layout.addWidget(self.ocr_bitmaps)

        # Performance Settings Section
        layout.addWidget(self._create_section_label("Performance Settings"))
        
        self.accelerator = QComboBox()
        self.accelerator.addItems(['auto', 'cpu', 'cuda', 'mps'])
        self.accelerator.setCurrentText('auto')
        
        self.workers = QComboBox()
        self.workers.addItems(['1', '2', '4', '8', '16'])
        self.workers.setCurrentText('4')
        
        perf_tooltips = {
            'Accelerator': 'Processing hardware:\n- auto: Best available\n- cpu: Compatible but slow\n- cuda: Fast (NVIDIA)\n- mps: Fast (Apple Silicon)',
            'Workers': 'Parallel processes. Higher = faster but more CPU/RAM (4-8 recommended)'
        }
        
        for label, combo, tooltip in [
            ('Accelerator:', self.accelerator, perf_tooltips['Accelerator']),
            ('Workers:', self.workers, perf_tooltips['Workers'])
        ]:
            self._add_labeled_widget(layout, label, combo, tooltip)

        # EasyOCR Settings
        layout.addWidget(self._create_section_label("EasyOCR Settings"))
        
        self.text_threshold = QComboBox()
        self.text_threshold.addItems(['0.5', '0.6', '0.7', '0.8', '0.9'])
        self.text_threshold.setCurrentText('0.7')
        
        self.low_text = QComboBox()
        self.low_text.addItems(['0.3', '0.35', '0.4', '0.45', '0.5'])
        self.low_text.setCurrentText('0.4')
        
        ocr_tooltips = {
            'Text Threshold': 'Main confidence filter. Higher (0.7+) = more reliable but might miss text',
            'Low Text': 'Faint text detection. Lower (0.3-0.4) = better for poor quality documents'
        }
        
        for label, combo, tooltip in [
            ('Text Threshold:', self.text_threshold, ocr_tooltips['Text Threshold']),
            ('Low Text:', self.low_text, ocr_tooltips['Low Text'])
        ]:
            self._add_labeled_widget(layout, label, combo, tooltip)

        # EasyOCR Advanced Settings
        layout.addWidget(self._create_section_label("EasyOCR Advanced Settings"))
        
        # Decoder selection
        self.decoder = QComboBox()
        self.decoder.addItems(['beamsearch', 'greedy', 'wordbeamsearch'])
        self.decoder.setCurrentText('beamsearch')
        
        # Batch size
        self.batch_size = QComboBox()
        self.batch_size.addItems(['1', '2', '4', '8', '16'])
        self.batch_size.setCurrentText('4')
        
        # Contrast settings
        self.contrast_ths = QComboBox()
        self.contrast_ths.addItems(['0.1', '0.2', '0.3', '0.4', '0.5'])
        self.contrast_ths.setCurrentText('0.1')
        
        self.adjust_contrast = QComboBox()
        self.adjust_contrast.addItems(['0.3', '0.4', '0.5', '0.6', '0.7'])
        self.adjust_contrast.setCurrentText('0.5')
        
        # Text detection settings
        self.link_threshold = QComboBox()
        self.link_threshold.addItems(['0.2', '0.3', '0.4', '0.5', '0.6'])
        self.link_threshold.setCurrentText('0.4')
        
        self.canvas_size = QComboBox()
        self.canvas_size.addItems(['1280', '2560', '3840', '5120'])
        self.canvas_size.setCurrentText('2560')
        
        self.mag_ratio = QComboBox()
        self.mag_ratio.addItems(['1.0', '1.25', '1.5', '1.75', '2.0'])
        self.mag_ratio.setCurrentText('1.5')
        
        # Bounding box settings
        self.slope_ths = QComboBox()
        self.slope_ths.addItems(['0.0', '0.025', '0.05', '0.075', '0.1'])
        self.slope_ths.setCurrentText('0.05')
        
        self.ycenter_ths = QComboBox()
        self.ycenter_ths.addItems(['0.1', '0.2', '0.3', '0.4', '0.5'])
        self.ycenter_ths.setCurrentText('0.3')
        
        self.height_ths = QComboBox()
        self.height_ths.addItems(['0.1', '0.2', '0.3', '0.4', '0.5'])
        self.height_ths.setCurrentText('0.3')
        
        self.width_ths = QComboBox()
        self.width_ths.addItems(['0.1', '0.2', '0.3', '0.4', '0.5'])
        self.width_ths.setCurrentText('0.3')

        advanced_tooltips = {
            'Decoder': 'Choose decoding method:\n- beamsearch: Best accuracy, slower\n- greedy: Faster but less accurate\n- wordbeamsearch: Balance of speed/accuracy',
            'Batch Size': 'Images processed at once. Higher = faster but more memory (4-8 recommended)',
            'Contrast Threshold': 'Lower values (0.1-0.3) detect low-contrast text, higher values reduce noise',
            'Adjust Contrast': 'Higher values (0.5-0.7) enhance faint text, lower values preserve original',
            'Link Threshold': 'Lower values (0.2-0.3) better connect broken text, higher values separate text',
            'Canvas Size': 'Image processing size. Larger = better for high-res, but slower',
            'Magnification': 'Text scaling. Higher values (1.5+) better for small text, but slower',
            'Slope Threshold': 'Higher values allow more tilted text merging. 0.05 = balanced',
            'Y-Center Threshold': 'Higher values merge more vertical text. 0.3 = good for paragraphs',
            'Height Threshold': 'Higher values merge different size text. 0.3 = good for headers/body',
            'Width Threshold': 'Higher values merge varying width text. 0.3 = balanced setting'
        }

        # Add all advanced options to the layout
        for label, combo, tooltip in [
            ('Decoder:', self.decoder, advanced_tooltips['Decoder']),
            ('Batch Size:', self.batch_size, advanced_tooltips['Batch Size']),
            ('Contrast Threshold:', self.contrast_ths, advanced_tooltips['Contrast Threshold']),
            ('Adjust Contrast:', self.adjust_contrast, advanced_tooltips['Adjust Contrast']),
            ('Link Threshold:', self.link_threshold, advanced_tooltips['Link Threshold']),
            ('Canvas Size:', self.canvas_size, advanced_tooltips['Canvas Size']),
            ('Magnification:', self.mag_ratio, advanced_tooltips['Magnification']),
            ('Slope Threshold:', self.slope_ths, advanced_tooltips['Slope Threshold']),
            ('Y-Center Threshold:', self.ycenter_ths, advanced_tooltips['Y-Center Threshold']),
            ('Height Threshold:', self.height_ths, advanced_tooltips['Height Threshold']),
            ('Width Threshold:', self.width_ths, advanced_tooltips['Width Threshold'])
        ]:
            self._add_labeled_widget(layout, label, combo, tooltip)

    def _create_section_label(self, text: str) -> QLabel:
        """Create a formatted section label with description."""
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Main label
        label = QLabel(text)
        label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
                font-size: 14px;
                padding: 5px 0px;
            }
        """)
        
        # Description text based on section
        description = QLabel()
        description.setWordWrap(True)
        description.setStyleSheet("color: #7f8c8d; font-size: 12px;")
        
        descriptions = {
            "Basic Options": "Configure basic document processing settings",
            "OCR Settings": "Set up language and general OCR behavior",
            "Performance Settings": "Optimize processing speed and resource usage",
            "EasyOCR Settings": "Basic OCR quality settings for text detection",
            "EasyOCR Advanced Settings": "Fine-tune OCR behavior for better accuracy"
        }
        
        if text in descriptions:
            description.setText(descriptions[text])
        
        layout.addWidget(label)
        layout.addWidget(description)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 5, 0, 5)
        
        return container

    def _add_labeled_widget(self, layout: QVBoxLayout, label_text: str, widget: QWidget, tooltip: str) -> None:
        """Add a labeled widget with tooltip to the layout."""
        h_layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        widget.setToolTip(tooltip)
        h_layout.addWidget(label)
        h_layout.addWidget(widget)
        layout.addLayout(h_layout)

    def _setup_progress_section(self, layout: QVBoxLayout) -> None:
        """Setup progress display section."""
        self.progress_label = QLabel()
        layout.addWidget(self.progress_label)

    def _setup_control_buttons(self, layout: QVBoxLayout) -> None:
        """Setup control buttons."""
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton('Start Processing')
        self.start_btn.clicked.connect(self.start_processing)
        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

    def select_input_folder(self) -> None:
        """Handle input folder selection."""
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_path.setText(folder)
            logger.info(f"Input directory selected: {folder}")

    def select_output_folder(self) -> None:
        """Handle output folder selection."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_path.setText(folder)
            logger.info(f"Output directory selected: {folder}")

    def start_processing(self) -> None:
        """Start document processing."""
        if not self._validate_paths():
            return

        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")

        config = ProcessingConfig(
            # Basic settings
            input_path=self.input_path.text(),
            output_path=self.output_path.text(),
            export_format=self.format_combo.currentText(),
            table_mode=self.table_mode.currentText(),
            force_ocr=self.force_ocr.isChecked(),
            ocr_bitmaps=self.ocr_bitmaps.isChecked(),
            temp_dir=self.temp_dir,
            
            # Performance settings
            accelerator=self.accelerator.currentText(),
            workers=int(self.workers.currentText()),
            
            # OCR settings
            ocr_languages=self.language_input.text().strip(),
            text_threshold=float(self.text_threshold.currentText()),
            low_text=float(self.low_text.currentText()),
            
            # Advanced EasyOCR settings
            decoder=self.decoder.currentText(),
            batch_size=int(self.batch_size.currentText()),
            contrast_ths=float(self.contrast_ths.currentText()),
            adjust_contrast=float(self.adjust_contrast.currentText()),
            link_threshold=float(self.link_threshold.currentText()),
            canvas_size=int(self.canvas_size.currentText()),
            mag_ratio=float(self.mag_ratio.currentText()),
            
            # Bounding box settings
            slope_ths=float(self.slope_ths.currentText()),
            ycenter_ths=float(self.ycenter_ths.currentText()),
            height_ths=float(self.height_ths.currentText()),
            width_ths=float(self.width_ths.currentText())
        )

        # Add validation and user feedback
        try:
            self.worker = DoclingWorker(config)
            self.worker.progress.connect(self._update_progress)
            self.worker.error.connect(self._handle_error)
            self.worker.finished.connect(self._processing_finished)

            self.start_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.progress_label.setText("Starting processing with EasyOCR...")
            self.worker.start()
            
        except ValueError as e:
            self.progress_label.setText(f"Configuration error: {str(e)}")
            logger.error(f"Configuration error: {str(e)}")

    def _validate_paths(self) -> bool:
        """Validate input and output paths."""
        if self.input_path.text() == 'No folder selected':
            self.progress_label.setText('Please select an input folder')
            return False
        if self.output_path.text() == 'No folder selected':
            self.progress_label.setText('Please select an output folder')
            return False
        return True

    def cancel_processing(self) -> None:
        """Cancel ongoing processing."""
        if self.worker:
            self.worker.cancel()
            self.progress_label.setText("Cancelling...")
            self.cancel_btn.setEnabled(False)

    def _update_progress(self, message: str) -> None:
        """Update progress display."""
        self.progress_label.setText(message)

    def _handle_error(self, error_message: str) -> None:
        """Handle processing errors."""
        self.progress_label.setText(f"Error: {error_message}")

    def _processing_finished(self) -> None:
        """Clean up after processing is finished."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Removed temporary directory: {self.temp_dir}")

        # Clean up environment variables
        if 'EASYOCR_TEXT_THRESHOLD' in os.environ:
            del os.environ['EASYOCR_TEXT_THRESHOLD']
        if 'EASYOCR_LOW_TEXT' in os.environ:
            del os.environ['EASYOCR_LOW_TEXT']

        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText("Processing completed")

def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 