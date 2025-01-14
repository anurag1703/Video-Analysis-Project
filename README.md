# Shoppable Item Extractor

This application extracts clothing items from uploaded videos using object detection. It utilizes the YOLOv8 model for efficient and accurate identification of clothing within video frames.

## Features

*   **Video Upload:** Users can upload videos in MP4 or MOV format.
*   **Frame Selection:** Efficiently selects representative frames from the video using a combination of uniform sampling and motion detection.
*   **Object Detection:** Employs YOLOv8 to detect and classify clothing items within the selected frames.
*   **Duplicate Item Removal:** Ensures that each unique clothing item is displayed only once, even if it appears in multiple frames.
*   **Image Display:** Displays the extracted clothing items with their class labels and confidence scores.
*   **Temporary File Handling:** Uses temporary files for video processing, ensuring no permanent files are saved to the user's system.

## Technologies Used

*   Python
*   OpenCV (`cv2`)
*   NumPy (`numpy`)
*   Ultralytics YOLO (`ultralytics`)
*   Streamlit (`streamlit`)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/](https://github.com/)<your_username>/<your_repository_name>.git
    cd <your_repository_name>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```
    Create a `requirements.txt` file with the following content:
    ```
    opencv-python
    numpy
    ultralytics
    streamlit
    ```

## Usage

1.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

    (Replace `app.py` with the actual name of your Python script if it's different).

2.  Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3.  Upload a video file using the file uploader.

4.  The extracted clothing items will be displayed on the screen.
