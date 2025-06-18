import os
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# For PDF parsing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pypdf
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        logging.warning("PyPDF2/pypdf not available - PDF parsing disabled")

# For image extraction
try:
    from PIL import Image
    import io
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False
    logging.warning("Pillow not available - PDF image extraction disabled")

def validate_pdf_file(file_path: str) -> bool:
    """Validate that PDF file exists and is readable"""
    if not os.path.exists(file_path):
        raise ValueError(f"PDF file not found: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise ValueError(f"PDF file not readable: {file_path}")
    return True

def extract_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """Extract basic file metadata"""
    stat = os.stat(file_path)
    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_size": stat.st_size,
        "file_extension": ".pdf",
        "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "parsed_at": datetime.now().isoformat()
    }

def extract_pdf_text_content(pdf_reader) -> List[Dict[str, Any]]:
    """Extract text content from PDF pages"""
    text_content = []
    
    try:
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text.strip():
                    text_content.append({
                        "page_number": page_num + 1,
                        "text": text.strip(),
                        "text_length": len(text),
                        "word_count": len(text.split())
                    })
                    
            except Exception as e:
                logging.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                continue
                
    except Exception as e:
        logging.warning(f"Error extracting text content: {str(e)}")
        
    return text_content

def extract_pdf_metadata_content(pdf_reader) -> Dict[str, Any]:
    """Extract PDF metadata"""
    metadata = {}
    
    try:
        if hasattr(pdf_reader, 'metadata'):
            pdf_metadata = pdf_reader.metadata
            
            # Extract common metadata fields
            metadata_fields = [
                'Title', 'Author', 'Subject', 'Creator', 'Producer', 
                'CreationDate', 'ModDate', 'Keywords', 'Trapped'
            ]
            
            for field in metadata_fields:
                if field in pdf_metadata:
                    metadata[field.lower()] = str(pdf_metadata[field])
                    
    except Exception as e:
        logging.warning(f"Error extracting PDF metadata: {str(e)}")
        
    return metadata

def extract_pdf_images(pdf_reader) -> List[Dict[str, Any]]:
    """Extract images from PDF pages"""
    images = []
    
    if not IMAGE_AVAILABLE:
        return images
    
    try:
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                
                # Extract images from page
                if hasattr(page, 'images'):
                    for img_num, image in enumerate(page.images):
                        try:
                            img_data = {
                                "page_number": page_num + 1,
                                "image_number": img_num + 1,
                                "image_type": image.image_type,
                                "image_size": len(image.data),
                                "image_width": getattr(image, 'width', None),
                                "image_height": getattr(image, 'height', None),
                                "image_format": getattr(image, 'format', None)
                            }
                            
                            # Try to get image dimensions using PIL
                            try:
                                img = Image.open(io.BytesIO(image.data))
                                img_data["pil_width"] = img.width
                                img_data["pil_height"] = img.height
                                img_data["pil_mode"] = img.mode
                            except:
                                pass
                                
                            images.append(img_data)
                            
                        except Exception as e:
                            logging.warning(f"Error processing image {img_num + 1} on page {page_num + 1}: {str(e)}")
                            continue
                            
            except Exception as e:
                logging.warning(f"Error extracting images from page {page_num + 1}: {str(e)}")
                continue
                
    except Exception as e:
        logging.warning(f"Error extracting images: {str(e)}")
        
    return images

def extract_pdf_annotations(pdf_reader) -> List[Dict[str, Any]]:
    """Extract annotations from PDF pages"""
    annotations = []
    
    try:
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                
                # Extract annotations from page
                if hasattr(page, 'annotations'):
                    for ann_num, annotation in enumerate(page.annotations):
                        try:
                            ann_data = {
                                "page_number": page_num + 1,
                                "annotation_number": ann_num + 1,
                                "annotation_type": getattr(annotation, 'subtype', 'Unknown'),
                                "annotation_content": getattr(annotation, 'contents', ''),
                                "annotation_author": getattr(annotation, 'author', ''),
                                "annotation_date": getattr(annotation, 'date', ''),
                                "annotation_rect": getattr(annotation, 'rect', None)
                            }
                            
                            annotations.append(ann_data)
                            
                        except Exception as e:
                            logging.warning(f"Error processing annotation {ann_num + 1} on page {page_num + 1}: {str(e)}")
                            continue
                            
            except Exception as e:
                logging.warning(f"Error extracting annotations from page {page_num + 1}: {str(e)}")
                continue
                
    except Exception as e:
        logging.warning(f"Error extracting annotations: {str(e)}")
        
    return annotations

def detect_cad_content(text_content: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect CAD-related content in PDF text"""
    cad_indicators = {
        "autocad": ["autocad", "cad", "dwg", "dxf", "drawing", "blueprint"],
        "dimensions": ["dimension", "length", "width", "height", "radius", "diameter", "scale"],
        "technical": ["technical", "engineering", "architectural", "construction", "design"],
        "measurements": ["mm", "cm", "m", "inch", "feet", "ft", "meter", "millimeter"],
        "layers": ["layer", "layers", "level", "levels"],
        "views": ["front", "side", "top", "bottom", "isometric", "orthographic", "perspective"],
        "materials": ["steel", "concrete", "wood", "aluminum", "plastic", "material"],
        "components": ["component", "part", "assembly", "subassembly", "detail"]
    }
    
    cad_detection = {
        "is_cad_document": False,
        "confidence_score": 0.0,
        "detected_features": {},
        "total_matches": 0
    }
    
    all_text = " ".join([item["text"].lower() for item in text_content])
    total_matches = 0
    
    for category, keywords in cad_indicators.items():
        matches = sum(1 for keyword in keywords if keyword in all_text)
        cad_detection["detected_features"][category] = {
            "keywords_found": matches,
            "total_keywords": len(keywords),
            "match_ratio": matches / len(keywords) if keywords else 0
        }
        total_matches += matches
    
    # Calculate confidence score
    total_keywords = sum(len(keywords) for keywords in cad_indicators.values())
    cad_detection["total_matches"] = total_matches
    cad_detection["confidence_score"] = total_matches / total_keywords if total_keywords > 0 else 0
    cad_detection["is_cad_document"] = cad_detection["confidence_score"] > 0.1
    
    return cad_detection

def parse_pdf_file(file_path: str) -> Dict[str, Any]:
    """
    Enhanced PDF file parser with comprehensive data extraction
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary containing parsed data with metadata
    """
    if not PDF_AVAILABLE:
        return {
            "status": "error",
            "file_metadata": extract_pdf_metadata(file_path),
            "error": "PDF parsing not available - PyPDF2/pypdf library required",
            "parsed_at": datetime.now().isoformat()
        }
    
    try:
        validate_pdf_file(file_path)
        metadata = extract_pdf_metadata(file_path)
        
        # Open PDF file
        with open(file_path, 'rb') as file:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
            except:
                # Try pypdf if PyPDF2 fails
                import pypdf
                file.seek(0)
                pdf_reader = pypdf.PdfReader(file)
        
        # Extract content
        text_content = extract_pdf_text_content(pdf_reader)
        pdf_metadata = extract_pdf_metadata_content(pdf_reader)
        images = extract_pdf_images(pdf_reader)
        annotations = extract_pdf_annotations(pdf_reader)
        
        # Detect CAD content
        cad_detection = detect_cad_content(text_content)
        
        # Create components
        components = []
        
        # Main document component
        main_component = {
            "component_id": f"PDF-{uuid.uuid4().hex[:8]}",
            "name": os.path.splitext(os.path.basename(file_path))[0],
            "type": "document",
            "properties": {
                "total_pages": len(pdf_reader.pages),
                "total_text_pages": len(text_content),
                "total_images": len(images),
                "total_annotations": len(annotations),
                "is_cad_document": cad_detection["is_cad_document"],
                "cad_confidence": cad_detection["confidence_score"],
                **pdf_metadata
            },
            "geometry": {
                "type": "document",
                "has_content": len(text_content) > 0 or len(images) > 0
            }
        }
        components.append(main_component)
        
        # Text content components
        for i, text_item in enumerate(text_content):
            text_component = {
                "component_id": f"PDF-TEXT-{uuid.uuid4().hex[:8]}",
                "name": f"Text_Page_{text_item['page_number']}",
                "type": "text_content",
                "properties": {
                    "page_number": text_item["page_number"],
                    "text_length": text_item["text_length"],
                    "word_count": text_item["word_count"],
                    "text_preview": text_item["text"][:200] + "..." if len(text_item["text"]) > 200 else text_item["text"]
                },
                "geometry": {
                    "type": "text",
                    "has_content": text_item["text_length"] > 0
                }
            }
            components.append(text_component)
        
        # Image components
        for i, image_item in enumerate(images):
            image_component = {
                "component_id": f"PDF-IMG-{uuid.uuid4().hex[:8]}",
                "name": f"Image_{image_item['page_number']}_{image_item['image_number']}",
                "type": "image",
                "properties": {
                    "page_number": image_item["page_number"],
                    "image_number": image_item["image_number"],
                    "image_type": image_item["image_type"],
                    "image_size": image_item["image_size"],
                    "image_width": image_item.get("image_width"),
                    "image_height": image_item.get("image_height"),
                    "image_format": image_item.get("image_format")
                },
                "geometry": {
                    "type": "image",
                    "has_dimensions": image_item.get("image_width") is not None
                }
            }
            components.append(image_component)
        
        # Annotation components
        for i, ann_item in enumerate(annotations):
            ann_component = {
                "component_id": f"PDF-ANN-{uuid.uuid4().hex[:8]}",
                "name": f"Annotation_{ann_item['page_number']}_{ann_item['annotation_number']}",
                "type": "annotation",
                "properties": {
                    "page_number": ann_item["page_number"],
                    "annotation_number": ann_item["annotation_number"],
                    "annotation_type": ann_item["annotation_type"],
                    "annotation_content": ann_item["annotation_content"],
                    "annotation_author": ann_item["annotation_author"],
                    "annotation_date": ann_item["annotation_date"]
                },
                "geometry": {
                    "type": "annotation",
                    "has_content": len(ann_item["annotation_content"]) > 0
                }
            }
            components.append(ann_component)
        
        return {
            "status": "success",
            "file_metadata": metadata,
            "pdf_metadata": pdf_metadata,
            "cad_detection": cad_detection,
            "statistics": {
                "total_components": len(components),
                "total_pages": len(pdf_reader.pages),
                "text_pages": len(text_content),
                "images_count": len(images),
                "annotations_count": len(annotations),
                "total_text_length": sum(item["text_length"] for item in text_content),
                "total_word_count": sum(item["word_count"] for item in text_content)
            },
            "components": components,
            "text_content": text_content,
            "images": images,
            "annotations": annotations,
            "parsed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error parsing PDF file {file_path}: {str(e)}")
        return {
            "status": "error",
            "file_metadata": extract_pdf_metadata(file_path),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "parsed_at": datetime.now().isoformat()
        }

# Legacy function for backward compatibility
def parse_pdf(content):
    """Legacy function - use parse_pdf_file instead"""
    if isinstance(content, str):
        return parse_pdf_file(content)
    else:
        return [{"name": "Parsed from PDF", "shape": "unknown"}]
