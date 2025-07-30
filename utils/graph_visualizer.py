import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple
import random
import math

class GraphVisualizer:
    """Utility for creating graph visualizations"""
    
    def __init__(self):
        self.color_palette = {
            'Employee': '#FF6B6B',
            'Department': '#4ECDC4',
            'Policy': '#45B7D1',
            'Document': '#96CEB4',
            'Unknown': '#FECA57'
        }
    
    def create_network_graph(self, nodes: List[Dict[str, Any]], 
                           edges: List[Dict[str, Any]]) -> go.Figure:
        """
        Create an interactive network graph using Plotly
        
        Args:
            nodes: List of node dictionaries with id, label, type
            edges: List of edge dictionaries with from, to, label
            
        Returns:
            Plotly figure object
        """
        # Create NetworkX graph for layout calculation
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node['id'], **node)
        
        # Add edges
        for edge in edges:
            G.add_edge(edge['from'], edge['to'], **edge)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in edges:
            x0, y0 = pos[edge['from']]
            x1, y1 = pos[edge['to']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(edge.get('label', ''))
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in nodes:
            x, y = pos[node['id']]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node.get('label', node['id']))
            
            # Color by type
            node_type = node.get('type', 'Unknown')
            node_colors.append(self.color_palette.get(node_type, self.color_palette['Unknown']))
            
            # Size by degree (number of connections)
            degree = G.degree(node['id'])
            node_sizes.append(max(20, min(60, 20 + degree * 5)))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Knowledge Graph Visualization',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Interactive Graph - Drag nodes to explore",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    def create_hierarchy_graph(self, org_data: Dict[str, Any]) -> go.Figure:
        """
        Create organizational hierarchy visualization
        
        Args:
            org_data: Organizational structure data
            
        Returns:
            Plotly figure object
        """
        # Create tree layout
        G = nx.DiGraph()
        
        # Add nodes and edges based on reporting structure
        for dept, dept_info in org_data.items():
            # Add department node
            dept_id = f"dept_{dept}"
            G.add_node(dept_id, label=dept, type='Department', level=0)
            
            # Add manager
            manager = dept_info.get('manager')
            if manager:
                manager_id = manager['id']
                G.add_node(manager_id, label=manager['name'], type='Manager', level=1)
                G.add_edge(dept_id, manager_id, label='manages')
                
                # Add employees
                for emp in dept_info.get('employees', []):
                    if emp['id'] != manager_id:
                        G.add_node(emp['id'], label=emp['name'], type='Employee', level=2)
                        G.add_edge(manager_id, emp['id'], label='supervises')
        
        # Use hierarchical layout
        pos = self._hierarchical_layout(G)
        
        # Create traces
        edge_trace = self._create_edge_trace(G, pos)
        node_trace = self._create_node_trace(G, pos)
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Organizational Hierarchy',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    def create_centrality_chart(self, centrality_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create centrality analysis chart
        
        Args:
            centrality_data: List of nodes with centrality scores
            
        Returns:
            Plotly figure object
        """
        if not centrality_data:
            return go.Figure()
        
        nodes = [item['node'] for item in centrality_data]
        scores = [item['centrality'] for item in centrality_data]
        
        fig = go.Figure(data=[
            go.Bar(
                x=nodes,
                y=scores,
                marker_color='lightblue',
                text=[f"{score:.3f}" for score in scores],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Node Centrality Analysis',
            xaxis_title='Nodes',
            yaxis_title='Centrality Score',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_relationship_pie_chart(self, relationship_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create pie chart for relationship types
        
        Args:
            relationship_data: List of relationship types with counts
            
        Returns:
            Plotly figure object
        """
        if not relationship_data:
            return go.Figure()
        
        labels = [item['relationship_type'] for item in relationship_data]
        values = [item['count'] for item in relationship_data]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3
        )])
        
        fig.update_layout(
            title='Relationship Type Distribution',
            annotations=[dict(text='Relationships', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig
    
    def create_similarity_heatmap(self, similarity_matrix: List[List[float]], 
                                labels: List[str]) -> go.Figure:
        """
        Create similarity heatmap
        
        Args:
            similarity_matrix: 2D matrix of similarity scores
            labels: Labels for rows and columns
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title='Document Similarity Heatmap',
            xaxis_title='Documents',
            yaxis_title='Documents'
        )
        
        return fig
    
    def create_performance_timeline(self, performance_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create performance timeline chart
        
        Args:
            performance_data: List of performance metrics over time
            
        Returns:
            Plotly figure object
        """
        if not performance_data:
            return go.Figure()
        
        timestamps = [item['timestamp'] for item in performance_data]
        response_times = [item['response_time'] for item in performance_data]
        confidence_scores = [item.get('confidence_score', 0) for item in performance_data]
        
        fig = go.Figure()
        
        # Add response time trace
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=response_times,
            mode='lines+markers',
            name='Response Time (s)',
            yaxis='y'
        ))
        
        # Add confidence score trace
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=confidence_scores,
            mode='lines+markers',
            name='Confidence Score',
            yaxis='y2'
        ))
        
        # Update layout for dual y-axis
        fig.update_layout(
            title='Performance Metrics Over Time',
            xaxis_title='Time',
            yaxis=dict(
                title='Response Time (seconds)',
                side='left'
            ),
            yaxis2=dict(
                title='Confidence Score',
                side='right',
                overlaying='y'
            ),
            legend=dict(x=0.01, y=0.99)
        )
        
        return fig
    
    def _hierarchical_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout for directed graph"""
        
        pos = {}
        levels = {}
        
        # Assign levels based on node attributes or topology
        for node, data in G.nodes(data=True):
            level = data.get('level', 0)
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        # Position nodes
        for level, nodes in levels.items():
            y = -level  # Higher levels at top
            for i, node in enumerate(nodes):
                x = i - len(nodes) / 2  # Center nodes horizontally
                pos[node] = (x, y)
        
        return pos
    
    def _create_edge_trace(self, G: nx.Graph, pos: Dict[str, Tuple[float, float]]) -> go.Scatter:
        """Create edge trace for graph visualization"""
        
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
    
    def _create_node_trace(self, G: nx.Graph, pos: Dict[str, Tuple[float, float]]) -> go.Scatter:
        """Create node trace for graph visualization"""
        
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(data.get('label', node))
            
            # Color by type
            node_type = data.get('type', 'Unknown')
            node_colors.append(self.color_palette.get(node_type, self.color_palette['Unknown']))
            
            # Size by degree
            degree = G.degree(node)
            node_sizes.append(max(20, min(60, 20 + degree * 5)))
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
    
    def create_query_analytics_dashboard(self, analytics_data: Dict[str, Any]) -> List[go.Figure]:
        """
        Create multiple charts for query analytics dashboard
        
        Args:
            analytics_data: Dictionary containing various analytics data
            
        Returns:
            List of Plotly figure objects
        """
        figures = []
        
        # Query frequency chart
        if 'query_frequency' in analytics_data:
            freq_data = analytics_data['query_frequency']
            fig_freq = go.Figure(data=[
                go.Bar(
                    x=list(freq_data.keys()),
                    y=list(freq_data.values()),
                    marker_color='lightcoral'
                )
            ])
            fig_freq.update_layout(
                title='Most Frequent Queries',
                xaxis_title='Query',
                yaxis_title='Frequency',
                xaxis_tickangle=-45
            )
            figures.append(fig_freq)
        
        # Query categories pie chart
        if 'categories' in analytics_data:
            cat_data = analytics_data['categories']
            fig_cat = go.Figure(data=[go.Pie(
                labels=list(cat_data.keys()),
                values=list(cat_data.values())
            )])
            fig_cat.update_layout(title='Query Categories Distribution')
            figures.append(fig_cat)
        
        return figures
    
    def create_knowledge_source_chart(self, source_data: Dict[str, int]) -> go.Figure:
        """
        Create chart showing knowledge source distribution
        
        Args:
            source_data: Dictionary mapping source names to counts
            
        Returns:
            Plotly figure object
        """
        if not source_data:
            return go.Figure()
        
        fig = go.Figure(data=[go.Pie(
            labels=list(source_data.keys()),
            values=list(source_data.values()),
            hole=0.4
        )])
        
        fig.update_layout(
            title='Knowledge Source Distribution',
            annotations=[dict(text='Sources', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        return fig

