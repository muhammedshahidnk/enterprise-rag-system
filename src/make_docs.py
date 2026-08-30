"""
Generates 6 short synthetic technical PDFs to act as our RAG knowledge base.
In real use you'd point this at your own manuals/SOPs — this is just so the
pipeline has something real to chew on.
"""
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

OUT = "documents"
os.makedirs(OUT, exist_ok=True)
styles = getSampleStyleSheet()

def make_pdf(filename, pages):
    """pages = list of strings, one per PDF page"""
    doc = SimpleDocTemplate(os.path.join(OUT, filename), pagesize=A4)
    flow = []
    for i, page_text in enumerate(pages):
        for para in page_text.strip().split("\n\n"):
            flow.append(Paragraph(para.strip().replace("\n", "<br/>"), styles["Normal"]))
            flow.append(Spacer(1, 10))
        if i != len(pages) - 1:
            from reportlab.platypus import PageBreak
            flow.append(PageBreak())
    doc.build(flow)
    print(f"Created {filename} ({len(pages)} pages)")

# 1. CNC Machine Manual
make_pdf("cnc_machine_manual.pdf", [
"""CNC Milling Machine Model X200 - Operation Manual, Section 1: Startup

Before powering on the X200, verify that the emergency stop button is in the
released (out) position. Check coolant reservoir level; it must be above the
minimum fill line marked on the tank. Power on the main disconnect switch
located at the rear of the machine, then wait for the control panel to
complete its boot sequence, which takes approximately 45 seconds.""",
"""CNC Milling Machine Model X200 - Section 2: Homing Procedure

After startup, the machine must be homed before any program can run. Press
the HOME button on the control panel. All three axes (X, Y, Z) will move to
their reference switches automatically. Do not open the enclosure door during
homing. If homing fails, check for obstructions near the limit switches and
retry. The homing sequence takes about 20 seconds.""",
"""CNC Milling Machine Model X200 - Section 3: Tool Change and Maintenance

Tool changes should only be performed with the spindle fully stopped. Use the
manual tool release lever on the spindle head, remove the old tool, insert
the new tool until it seats fully, and release the lever to lock it in place.
Perform spindle lubrication every 500 operating hours using ISO VG32 oil.
Failure to lubricate on schedule voids the warranty on the spindle bearing.""",
])

# 2. Quality SOP
make_pdf("quality_control_sop.pdf", [
"""Standard Operating Procedure QC-114: Incoming Material Inspection

Purpose: This SOP defines the inspection process for raw material received
from approved suppliers. All incoming lots must be sampled according to
ANSI/ASQ Z1.4 general inspection level II before being released to
production. Rejected lots must be quarantined in the red-tag area within
2 hours of failed inspection.""",
"""SOP QC-114: Sampling and Acceptance Criteria

Sample size is determined by lot size per the AQL table, using an acceptable
quality level of 1.0 for critical dimensions and 2.5 for cosmetic defects.
If the number of defective units in the sample exceeds the acceptance number,
the entire lot is rejected. All measurements must be recorded on Form QC-114A
and retained for 3 years for audit traceability.""",
"""SOP QC-114: Non-Conformance Handling

Any non-conforming material identified during inspection must be logged in
the Non-Conformance Report (NCR) system within the same shift. The quality
engineer on duty determines disposition: use-as-is, rework, return-to-vendor,
or scrap. Disposition decisions require sign-off from both the quality
manager and the production supervisor before material can be moved.""",
])

# 3. Computer Vision docs
make_pdf("opencv_basics_guide.pdf", [
"""Computer Vision Fundamentals: Image Preprocessing

Before running detection models, images are typically converted to grayscale
to reduce computational load, then a Gaussian blur is applied to suppress
high-frequency noise. Common kernel sizes are 3x3 or 5x5. Histogram
equalization can improve contrast in poorly lit images and is especially
useful for industrial inspection cameras with inconsistent lighting.""",
"""Computer Vision Fundamentals: Edge Detection and Contours

The Canny edge detector is the standard method for edge detection, using two
threshold values to classify strong and weak edges. After edge detection,
contours can be extracted using findContours, which returns a hierarchy of
boundary curves. Contour area and perimeter are commonly used to filter out
noise blobs from real objects of interest.""",
"""Computer Vision Fundamentals: Object Detection Pipelines

Modern object detection pipelines (YOLO, Faster R-CNN) output bounding boxes
with class labels and confidence scores. A confidence threshold of 0.5 is a
common default, but should be tuned per use case. Non-maximum suppression
(NMS) removes duplicate overlapping boxes for the same object, typically
using an IoU threshold of 0.4 to 0.5.""",
])

# 4. Python logging docs (mimicking official-style docs)
make_pdf("python_logging_reference.pdf", [
"""Python Logging Module Reference: Basic Usage

The logging module provides a flexible framework for emitting log messages
from Python programs. The five standard levels, in increasing severity, are
DEBUG, INFO, WARNING, ERROR, and CRITICAL. Use logging.basicConfig() to
configure the root logger's level and output format before any log calls
are made elsewhere in the program.""",
"""Python Logging Module Reference: Handlers and Formatters

Handlers determine where log records go: StreamHandler sends them to
console, FileHandler writes them to a file, and RotatingFileHandler rotates
log files once they reach a configured size. Each handler can have its own
Formatter, allowing different destinations to show different levels of
detail from the same log call.""",
"""Python Logging Module Reference: Best Practices

Avoid using the root logger directly in library code; instead call
logging.getLogger(__name__) so consumers of your library can control
verbosity independently. Never log sensitive data such as passwords or API
keys. Use lazy formatting, e.g. logger.info("value: %s", x), instead of
f-strings, to avoid the formatting cost when the log level suppresses it.""",
])

# 5. REST API guide
make_pdf("internal_rest_api_guide.pdf", [
"""Internal REST API Guide: Authentication

All API requests must include a Bearer token in the Authorization header.
Tokens are issued by the /auth/token endpoint and expire after 60 minutes.
Refresh tokens are valid for 30 days and can be exchanged for a new access
token via the /auth/refresh endpoint. Rate limiting is enforced at 100
requests per minute per API key.""",
"""Internal REST API Guide: Error Handling

The API returns standard HTTP status codes. A 429 status indicates rate
limiting has been triggered; clients should implement exponential backoff
starting at 1 second. A 503 indicates the service is temporarily
unavailable, typically during deployments, and requests should be retried
after 30 seconds.""",
])

# 6. Database backup procedure
make_pdf("database_backup_procedure.pdf", [
"""Database Backup and Recovery Procedure

Full backups of the production database are taken nightly at 02:00 UTC and
retained for 14 days. Incremental backups run every 4 hours. Backup
integrity is verified weekly by restoring to a staging environment and
running a checksum comparison against the source. Recovery Point Objective
(RPO) is 4 hours and Recovery Time Objective (RTO) is 2 hours.""",
"""Database Backup and Recovery Procedure: Restore Steps

To restore from backup, first stop all application connections to the
database. Locate the most recent full backup and any incremental backups
taken after it. Apply the full backup first, then apply incrementals in
chronological order. After restore, run the validation script
validate_restore.sql before reopening connections to applications.""",
])

print("\nAll documents created in ./documents/")
