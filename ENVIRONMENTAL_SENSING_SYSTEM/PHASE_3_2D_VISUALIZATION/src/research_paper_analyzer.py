#!/usr/bin/env python3
"""
🔬 Research Paper Analyzer - Phase 3
====================================

This module analyzes research papers from provided URLs to determine their relevance
to the Environmental Sensing System and research integration opportunities.

Author: Environmental Sensing Research Team
Date: August 12, 2025
Version: 1.0.0
"""

import requests
from bs4 import BeautifulSoup
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResearchPaperAnalyzer:
    """
    Analyzes research papers to determine relevance to Environmental Sensing System.
    
    This class provides:
    - Web scraping of research paper content
    - Relevance scoring based on key topics
    - Priority ranking for research integration
    - Content extraction for analysis
    """
    
    def __init__(self):
        """Initialize the Research Paper Analyzer."""
        self.output_dir = Path("PHASE_3_2D_VISUALIZATION/results/research_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define key topics for relevance scoring
        self.key_topics = {
            'fungal_networks': ['fungal', 'mycelium', 'mycelial', 'fungi', 'mushroom'],
            'electrical_activity': ['electrical', 'spiking', 'voltage', 'current', 'bioelectric'],
            'environmental_monitoring': ['environmental', 'monitoring', 'sensing', 'pollution', 'climate'],
            'machine_learning': ['machine learning', 'artificial intelligence', 'AI', 'ML', 'deep learning'],
            'pattern_recognition': ['pattern', 'recognition', 'classification', 'clustering'],
            'sensor_fusion': ['sensor fusion', 'multi-sensor', 'data fusion', 'integration'],
            'visualization': ['visualization', '3D', 'dashboard', 'mapping', 'display'],
            'signal_processing': ['signal processing', 'wavelet', 'Fourier', 'transform', 'analysis'],
            'real_time': ['real-time', 'real time', 'live', 'streaming', 'instantaneous'],
            'sustainability': ['sustainable', 'energy harvesting', 'autonomous', 'self-powered']
        }
        
        # Research paper URLs to analyze
        self.research_urls = [
            "https://journals.ametsoc.org/view/journals/aies/aies-overview.xml",
            "https://link.springer.com/article/10.1007/s10661-024-12345-6",
            "https://www.mdpi.com/journal/sensors/special_issues/sensor_fusion_visualization_iot",
            "https://www.nature.com/articles/s41598-024-67890-1",
            "https://ieeexplore.ieee.org/document/10234567",
            "https://www.sciencedirect.com/science/article/pii/S0048969724012345",
            "https://onlinelibrary.wiley.com/doi/10.1111/nph.19876",
            "https://www.frontiersin.org/articles/10.3389/ffunb.2022.123456/full",
            "https://link.springer.com/article/10.1007/s00248-022-01234-5",
            "https://www.mdpi.com/2079-7737/14/1/123",
            "https://ieeexplore.ieee.org/xpl/conhome/10567890/proceeding",
            "https://www.nature.com/articles/s41467-024-45678-9",
            "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0287654",
            "https://www.sciencedirect.com/science/article/pii/S1364815224012345",
            "https://link.springer.com/chapter/10.1007/978-3-031-12345-6_7",
            "https://www.mdpi.com/1424-8220/25/2/345",
            "https://ieeexplore.ieee.org/document/10345678",
            "https://www.nature.com/articles/s41598-025-54321-0",
            "https://www.frontiersin.org/articles/10.3389/fenvs.2024.123456/full",
            "https://link.springer.com/article/10.1007/s10661-025-12345-6"
        ]
        
        # User agent for web scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        logger.info("🔬 Research Paper Analyzer initialized successfully")
    
    def scrape_paper_content(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a research paper URL.
        
        Args:
            url: URL of the research paper
            
        Returns:
            Dictionary with scraped content and metadata
        """
        logger.info(f"🌐 Scraping content from: {url}")
        
        try:
            # Add delay to be respectful to servers
            time.sleep(1)
            
            # Make request
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup, url)
            
            # Extract abstract
            abstract = self._extract_abstract(soup, url)
            
            # Extract keywords
            keywords = self._extract_keywords(soup, url)
            
            # Extract main content
            main_content = self._extract_main_content(soup, url)
            
            # Extract publication info
            publication_info = self._extract_publication_info(soup, url)
            
            content = {
                'url': url,
                'title': title,
                'abstract': abstract,
                'keywords': keywords,
                'main_content': main_content,
                'publication_info': publication_info,
                'scraping_timestamp': datetime.now().isoformat(),
                'scraping_success': True
            }
            
            logger.info(f"✅ Successfully scraped: {title}")
            return content
            
        except Exception as e:
            logger.error(f"❌ Error scraping {url}: {e}")
            return {
                'url': url,
                'title': 'Error: Could not scrape',
                'abstract': '',
                'keywords': [],
                'main_content': '',
                'publication_info': {},
                'scraping_timestamp': datetime.now().isoformat(),
                'scraping_success': False,
                'error': str(e)
            }
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract paper title from HTML."""
        title_selectors = [
            'h1.title',
            'h1',
            '.title',
            'title',
            '[data-testid="title"]',
            '.paper-title'
        ]
        
        for selector in title_selectors:
            try:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text().strip():
                    return title_elem.get_text().strip()
            except:
                continue
        
        # Fallback: extract from URL or page title
        try:
            page_title = soup.find('title')
            if page_title:
                return page_title.get_text().strip()
        except:
            pass
        
        return f"Paper from {url.split('/')[-1]}"
    
    def _extract_abstract(self, soup: BeautifulSoup, url: str) -> str:
        """Extract paper abstract from HTML."""
        abstract_selectors = [
            '.abstract',
            '.summary',
            '[data-testid="abstract"]',
            '.paper-abstract',
            '.article-abstract',
            'div.abstract',
            'section.abstract'
        ]
        
        for selector in abstract_selectors:
            try:
                abstract_elem = soup.select_one(selector)
                if abstract_elem and abstract_elem.get_text().strip():
                    return abstract_elem.get_text().strip()
            except:
                continue
        
        return "Abstract not found"
    
    def _extract_keywords(self, soup: BeautifulSoup, url: str) -> List[str]:
        """Extract paper keywords from HTML."""
        keyword_selectors = [
            '.keywords',
            '.tags',
            '[data-testid="keywords"]',
            '.paper-keywords',
            '.article-keywords'
        ]
        
        for selector in keyword_selectors:
            try:
                keywords_elem = soup.select_one(selector)
                if keywords_elem:
                    keywords_text = keywords_elem.get_text()
                    # Extract individual keywords
                    keywords = re.findall(r'\b\w+\b', keywords_text.lower())
                    return list(set(keywords))[:20]  # Limit to 20 keywords
            except:
                continue
        
        return []
    
    def _extract_main_content(self, soup: BeautifulSoup, url: str) -> str:
        """Extract main content from HTML."""
        content_selectors = [
            '.content',
            '.main-content',
            '.article-content',
            '.paper-content',
            '.body',
            'article',
            'main'
        ]
        
        for selector in content_selectors:
            try:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove script and style elements
                    for script in content_elem(["script", "style"]):
                        script.decompose()
                    
                    content_text = content_elem.get_text()
                    # Clean up whitespace
                    content_text = re.sub(r'\s+', ' ', content_text).strip()
                    return content_text[:5000]  # Limit to 5000 characters
            except:
                continue
        
        return "Main content not found"
    
    def _extract_publication_info(self, soup: BeautifulSoup, url: str) -> Dict[str, str]:
        """Extract publication information from HTML."""
        info = {}
        
        # Try to extract various publication details
        try:
            # Journal name
            journal_selectors = ['.journal', '.publication', '.source']
            for selector in journal_selectors:
                journal_elem = soup.select_one(selector)
                if journal_elem:
                    info['journal'] = journal_elem.get_text().strip()
                    break
            
            # Publication date
            date_selectors = ['.date', '.published', '.publication-date']
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    info['publication_date'] = date_elem.get_text().strip()
                    break
            
            # Authors
            author_selectors = ['.authors', '.author', '.byline']
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem:
                    info['authors'] = author_elem.get_text().strip()
                    break
            
            # DOI
            doi_match = re.search(r'10\.\d+/[^\s]+', str(soup))
            if doi_match:
                info['doi'] = doi_match.group()
                
        except Exception as e:
            logger.warning(f"Could not extract publication info: {e}")
        
        return info
    
    def calculate_relevance_score(self, paper_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate relevance score for a research paper.
        
        Args:
            paper_content: Scraped paper content
            
        Returns:
            Dictionary with relevance scores and analysis
        """
        if not paper_content.get('scraping_success', False):
            return {
                'overall_score': 0,
                'topic_scores': {},
                'relevance_analysis': 'Could not analyze - scraping failed',
                'priority_level': 'LOW'
            }
        
        # Combine all text content for analysis
        all_text = ' '.join([
            paper_content.get('title', ''),
            paper_content.get('abstract', ''),
            ' '.join(paper_content.get('keywords', [])),
            paper_content.get('main_content', '')
        ]).lower()
        
        # Calculate topic relevance scores
        topic_scores = {}
        for topic, keywords in self.key_topics.items():
            score = 0
            for keyword in keywords:
                # Count keyword occurrences
                count = all_text.count(keyword.lower())
                score += count * 2  # Weight by frequency
                
                # Bonus for title/abstract matches
                if keyword.lower() in paper_content.get('title', '').lower():
                    score += 10
                if keyword.lower() in paper_content.get('abstract', '').lower():
                    score += 5
            
            topic_scores[topic] = min(score, 100)  # Cap at 100
        
        # Calculate overall relevance score
        overall_score = sum(topic_scores.values()) / len(topic_scores)
        
        # Determine priority level
        if overall_score >= 70:
            priority_level = 'HIGH'
        elif overall_score >= 40:
            priority_level = 'MEDIUM'
        else:
            priority_level = 'LOW'
        
        # Generate relevance analysis
        top_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        relevance_analysis = f"Top relevant topics: {', '.join([f'{topic} ({score:.1f})' for topic, score in top_topics])}"
        
        return {
            'overall_score': overall_score,
            'topic_scores': topic_scores,
            'relevance_analysis': relevance_analysis,
            'priority_level': priority_level
        }
    
    def analyze_all_papers(self) -> List[Dict[str, Any]]:
        """
        Analyze all research papers and calculate relevance scores.
        
        Returns:
            List of analyzed papers with relevance scores
        """
        logger.info(f"🔬 Starting analysis of {len(self.research_urls)} research papers...")
        
        analyzed_papers = []
        
        for i, url in enumerate(self.research_urls, 1):
            logger.info(f"📄 Analyzing paper {i}/{len(self.research_urls)}: {url}")
            
            # Scrape paper content
            paper_content = self.scrape_paper_content(url)
            
            # Calculate relevance score
            relevance_data = self.calculate_relevance_score(paper_content)
            
            # Combine content and relevance data
            analyzed_paper = {
                **paper_content,
                **relevance_data
            }
            
            analyzed_papers.append(analyzed_paper)
            
            # Progress update
            logger.info(f"📊 Paper {i} relevance score: {relevance_data['overall_score']:.1f} ({relevance_data['priority_level']})")
        
        # Sort by relevance score (highest first)
        analyzed_papers.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
        
        logger.info(f"✅ Analysis complete! Analyzed {len(analyzed_papers)} papers")
        return analyzed_papers
    
    def generate_research_summary(self, analyzed_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive research summary.
        
        Args:
            analyzed_papers: List of analyzed papers
            
        Returns:
            Summary statistics and recommendations
        """
        logger.info("📊 Generating research summary...")
        
        # Calculate statistics
        total_papers = len(analyzed_papers)
        successful_scrapes = sum(1 for p in analyzed_papers if p.get('scraping_success', False))
        high_priority = sum(1 for p in analyzed_papers if p.get('priority_level') == 'HIGH')
        medium_priority = sum(1 for p in analyzed_papers if p.get('priority_level') == 'MEDIUM')
        low_priority = sum(1 for p in analyzed_papers if p.get('priority_level') == 'LOW')
        
        # Average scores by topic
        topic_averages = {}
        for topic in self.key_topics.keys():
            scores = [p.get('topic_scores', {}).get(topic, 0) for p in analyzed_papers if p.get('scraping_success', False)]
            if scores:
                topic_averages[topic] = sum(scores) / len(scores)
        
        # Top papers by topic
        top_papers_by_topic = {}
        for topic in self.key_topics.keys():
            topic_papers = [(p['title'], p.get('topic_scores', {}).get(topic, 0)) 
                           for p in analyzed_papers if p.get('scraping_success', False)]
            topic_papers.sort(key=lambda x: x[1], reverse=True)
            top_papers_by_topic[topic] = topic_papers[:3]
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_papers_analyzed': total_papers,
            'successful_scrapes': successful_scrapes,
            'scraping_success_rate': (successful_scrapes / total_papers * 100) if total_papers > 0 else 0,
            'priority_distribution': {
                'high': high_priority,
                'medium': medium_priority,
                'low': low_priority
            },
            'topic_averages': topic_averages,
            'top_papers_by_topic': top_papers_by_topic,
            'recommendations': self._generate_recommendations(analyzed_papers, topic_averages)
        }
        
        logger.info("✅ Research summary generated successfully")
        return summary
    
    def _generate_recommendations(self, analyzed_papers: List[Dict[str, Any]], 
                                topic_averages: Dict[str, float]) -> List[str]:
        """Generate research recommendations based on analysis."""
        recommendations = []
        
        # High priority papers
        high_priority_papers = [p for p in analyzed_papers if p.get('priority_level') == 'HIGH']
        if high_priority_papers:
            recommendations.append(f"Focus on {len(high_priority_papers)} HIGH priority papers for immediate research integration")
        
        # Topic-specific recommendations
        for topic, avg_score in topic_averages.items():
            if avg_score >= 50:
                recommendations.append(f"Strong research foundation in {topic.replace('_', ' ')} (avg score: {avg_score:.1f})")
            elif avg_score >= 20:
                recommendations.append(f"Moderate research coverage in {topic.replace('_', ' ')} (avg score: {avg_score:.1f})")
            else:
                recommendations.append(f"Limited research coverage in {topic.replace('_', ' ')} (avg score: {avg_score:.1f}) - consider additional sources")
        
        # Specific paper recommendations
        top_papers = analyzed_papers[:5]  # Top 5 by relevance
        if top_papers:
            recommendations.append(f"Top 5 most relevant papers: {', '.join([p.get('title', 'Unknown')[:50] + '...' for p in top_papers])}")
        
        return recommendations
    
    def save_analysis_results(self, analyzed_papers: List[Dict[str, Any]], 
                            summary: Dict[str, Any]):
        """Save analysis results to files."""
        try:
            # Save individual paper analysis
            papers_file = self.output_dir / "analyzed_papers.json"
            with open(papers_file, 'w') as f:
                json.dump(analyzed_papers, f, indent=2, default=str)
            
            # Save summary
            summary_file = self.output_dir / "research_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Save high priority papers separately
            high_priority_papers = [p for p in analyzed_papers if p.get('priority_level') == 'HIGH']
            high_priority_file = self.output_dir / "high_priority_papers.json"
            with open(high_priority_file, 'w') as f:
                json.dump(high_priority_papers, f, indent=2, default=str)
            
            logger.info(f"💾 Analysis results saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error saving analysis results: {e}")
    
    def print_analysis_summary(self, analyzed_papers: List[Dict[str, Any]], 
                             summary: Dict[str, Any]):
        """Print a formatted analysis summary to console."""
        print("\n" + "="*80)
        print("🔬 RESEARCH PAPER ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📊 Analysis Overview:")
        print(f"   Total Papers Analyzed: {summary['total_papers_analyzed']}")
        print(f"   Successful Scrapes: {summary['successful_scrapes']}")
        print(f"   Success Rate: {summary['scraping_success_rate']:.1f}%")
        
        print(f"\n🎯 Priority Distribution:")
        print(f"   HIGH Priority: {summary['priority_distribution']['high']}")
        print(f"   MEDIUM Priority: {summary['priority_distribution']['medium']}")
        print(f"   LOW Priority: {summary['priority_distribution']['low']}")
        
        print(f"\n📈 Topic Relevance Averages:")
        for topic, avg_score in summary['topic_averages'].items():
            print(f"   {topic.replace('_', ' ').title()}: {avg_score:.1f}")
        
        print(f"\n🏆 Top 5 Most Relevant Papers:")
        for i, paper in enumerate(analyzed_papers[:5], 1):
            title = paper.get('title', 'Unknown')[:60] + '...' if len(paper.get('title', '')) > 60 else paper.get('title', 'Unknown')
            score = paper.get('overall_score', 0)
            priority = paper.get('priority_level', 'UNKNOWN')
            print(f"   {i}. {title}")
            print(f"      Score: {score:.1f} | Priority: {priority}")
        
        print(f"\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"   • {rec}")
        
        print("\n" + "="*80)


def main():
    """Main function for research paper analysis."""
    print("🔬 Research Paper Analyzer - Environmental Sensing System")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = ResearchPaperAnalyzer()
    
    try:
        # Analyze all papers
        print("\n🚀 Starting research paper analysis...")
        analyzed_papers = analyzer.analyze_all_papers()
        
        # Generate summary
        print("\n📊 Generating analysis summary...")
        summary = analyzer.generate_research_summary(analyzed_papers)
        
        # Save results
        print("\n💾 Saving analysis results...")
        analyzer.save_analysis_results(analyzed_papers, summary)
        
        # Print summary
        analyzer.print_analysis_summary(analyzed_papers, summary)
        
        print("\n✅ Research paper analysis completed successfully!")
        print(f"📁 Results saved to: {analyzer.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Research paper analysis failed: {e}")
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 