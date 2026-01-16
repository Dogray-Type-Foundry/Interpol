from __future__ import annotations
import objc
import os
import time
import math
from typing import Optional, Tuple, Dict, List, Any
from AppKit import (
    NSBezierPath, NSColor, NSPoint, NSMenu, NSMenuItem, NSApplication, NSString,
    NSView, NSSlider, NSTextField, NSAffineTransform, NSRectFill, NSWindow, NSGraphicsContext,
    NSBackingStoreBuffered, NSMakeRect, NSViewWidthSizable, NSViewHeightSizable,
    NSTitledWindowMask, NSResizableWindowMask, NSShadow,
    NSLayoutConstraint, NSLayoutAttributeHeight, NSLayoutRelationEqual, NSLayoutAttributeNotAnAttribute,
    NSImage, NSFont, NSFontAttributeName, NSForegroundColorAttributeName,
    NSAttributedString, NSMutableDictionary, NSRect, NSSize, NSControlSizeMini, NSControlSizeSmall,
    # Additional imports for direct NSPanel-based SyncRatiosPanel
    NSPanel, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable, NSWindowStyleMaskUtilityWindow,
    NSButton, NSButtonTypeSwitch, NSButtonTypeMomentaryPushIn, NSScreen, NSBezelStyleRounded,
    NSTextAlignmentRight, NSTextAlignmentCenter, NSTextAlignmentLeft
)
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
from fontTools.varLib.models import VariationModel, normalizeValue
from GlyphsApp import (
    DRAWBACKGROUND,
    GSControlLayer,
    UPDATEINTERFACE,
    WINDOW_MENU,
    Glyphs,
    GSEditViewController,
    GSSMOOTH,
    GSOFFCURVE,
)
from GlyphsApp.plugins import PalettePlugin, ReporterPlugin, SelectTool
from vanilla import EditText, Group, Slider, TextBox, Window, CheckBox, PopUpButton, Button, ComboBox, FloatingWindow, RadioGroup, ColorWell, HorizontalLine

KEY = "xyz.dogray.interpol"

from Foundation import NSUserDefaults, NSTimer, NSObject

# Debug flag - set to True to enable debug logging
DEBUG_INTERPOLATE = False

def debug_log(msg: str) -> None:
    """Print debug message if debugging is enabled.
    NOTE: For zero overhead when disabled, callers should check DEBUG_INTERPOLATE
    before constructing expensive log messages, or use debug_log_lazy() instead."""
    if DEBUG_INTERPOLATE:
        print(f"[Interpol] {msg}")

def debug_log_lazy(msg_func):
    """Lazy debug logging - msg_func is only called if debugging is enabled.
    Use this for expensive log messages: debug_log_lazy(lambda: f"expensive {computation}")"""
    if DEBUG_INTERPOLATE:
        print(f"[Interpol] {msg_func()}")


# =============================================================================
# Synchronization Helper - Ratio synchronization across masters
# =============================================================================

class SynchronizationHelper:
    """
    Helper class for synchronizing point ratios across masters to fix interpolation kinks.
    
    The ratio of a smooth point is defined as:
        ratio = distance(prev_point → current_point) / distance(current_point → next_point)
    
    When ratios differ across masters, the interpolated outline can develop kinks.
    This helper adjusts points to make ratios consistent across all masters.
    """
    
    # -------------------------------------------------------------------------
    # Bezier Curve Utilities for Curve Compensation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def bezier_point(p0, p1, p2, p3, t):
        """
        Evaluate a cubic Bezier curve at parameter t.
        Returns (x, y) tuple.
        """
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        t2 = t * t
        t3 = t2 * t
        
        x = mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0]
        y = mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1]
        return (x, y)
    
    @staticmethod
    def get_curve_midpoint(p0, p1, p2, p3):
        """Get the curve point at t=0.5 - this is what we want to preserve."""
        return SynchronizationHelper.bezier_point(p0, p1, p2, p3, 0.5)
    
    @staticmethod
    def distance(p1, p2):
        """Euclidean distance between two points."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    
    @staticmethod
    def find_handle_scale_for_midpoint(p0, p1_dir, p2_fixed, p3, target_midpoint, 
                                        scale_min=0.5, scale_max=1.5, steps=101):
        """
        Find the scale factor for p1 (relative to p0) that makes the curve 
        pass closest to target_midpoint at t=0.5.
        
        p1_dir is the direction vector from p0 toward original p1.
        p2_fixed is the other handle (already in final position).
        
        Returns the best scale factor.
        """
        best_scale = 1.0
        best_dist = float('inf')
        
        for i in range(steps):
            scale = scale_min + (scale_max - scale_min) * i / (steps - 1)
            p1_test = (p0[0] + p1_dir[0] * scale, p0[1] + p1_dir[1] * scale)
            mid = SynchronizationHelper.bezier_point(p0, p1_test, p2_fixed, p3, 0.5)
            dist = SynchronizationHelper.distance(mid, target_midpoint)
            if dist < best_dist:
                best_dist = dist
                best_scale = scale
        
        return best_scale
    
    @staticmethod
    def find_handle_scale_for_midpoint_p2(p0, p1_fixed, p2_dir, p3, target_midpoint,
                                           scale_min=0.5, scale_max=1.5, steps=101):
        """
        Find the scale factor for p2 (relative to p3) that makes the curve
        pass closest to target_midpoint at t=0.5.
        
        p2_dir is the direction vector from p3 toward original p2.
        p1_fixed is the other handle (already in final position).
        
        Returns the best scale factor.
        """
        best_scale = 1.0
        best_dist = float('inf')
        
        for i in range(steps):
            scale = scale_min + (scale_max - scale_min) * i / (steps - 1)
            p2_test = (p3[0] + p2_dir[0] * scale, p3[1] + p2_dir[1] * scale)
            mid = SynchronizationHelper.bezier_point(p0, p1_fixed, p2_test, p3, 0.5)
            dist = SynchronizationHelper.distance(mid, target_midpoint)
            if dist < best_dist:
                best_dist = dist
                best_scale = scale
        
        return best_scale
    
    @staticmethod
    def get_curve_segment_info(node, path, direction):
        """
        Get curve segment info for compensation.
        
        Args:
            node: The smooth on-curve node
            path: The path containing the node
            direction: "prev" or "next"
            
        Returns dict with:
            - p0, p1, p2, p3: Control points as (x, y) tuples
            - opp_handle: The opposite handle node (to be adjusted)
            - opp_oncurve_idx: Index of the opposite on-curve node
            - midpoint: Original curve midpoint at t=0.5
        Or None if not a curve segment.
        """
        nodes = list(path.nodes)
        num_nodes = len(nodes)
        node_idx = nodes.index(node)
        
        if direction == "prev":
            # Curve coming INTO this node: prev_oncurve -> opp_handle -> our_handle -> node
            our_handle = nodes[(node_idx - 1) % num_nodes]
            if our_handle.type != GSOFFCURVE:
                return None
            
            # Find previous on-curve
            prev_oncurve_idx = (node_idx - 2) % num_nodes
            while nodes[prev_oncurve_idx].type == GSOFFCURVE and prev_oncurve_idx != node_idx:
                prev_oncurve_idx = (prev_oncurve_idx - 1) % num_nodes
            
            prev_oncurve = nodes[prev_oncurve_idx]
            if prev_oncurve.type == GSOFFCURVE:
                return None
            
            # The opposite handle (from prev_oncurve side)
            opp_handle_idx = (prev_oncurve_idx + 1) % num_nodes
            opp_handle = nodes[opp_handle_idx]
            if opp_handle.type != GSOFFCURVE:
                return None  # Line segment, not curve
            
            p0 = (prev_oncurve.position.x, prev_oncurve.position.y)
            p1 = (opp_handle.position.x, opp_handle.position.y)
            p2 = (our_handle.position.x, our_handle.position.y)
            p3 = (node.position.x, node.position.y)
            
            return {
                'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3,
                'opp_handle': opp_handle,
                'opp_oncurve_idx': prev_oncurve_idx,
                'midpoint': SynchronizationHelper.get_curve_midpoint(p0, p1, p2, p3),
                'anchor': p0,  # p1 scales relative to p0
            }
        
        else:  # "next"
            # Curve going OUT from this node: node -> our_handle -> opp_handle -> next_oncurve
            our_handle = nodes[(node_idx + 1) % num_nodes]
            if our_handle.type != GSOFFCURVE:
                return None
            
            # Find next on-curve
            next_oncurve_idx = (node_idx + 2) % num_nodes
            while nodes[next_oncurve_idx].type == GSOFFCURVE and next_oncurve_idx != node_idx:
                next_oncurve_idx = (next_oncurve_idx + 1) % num_nodes
            
            next_oncurve = nodes[next_oncurve_idx]
            if next_oncurve.type == GSOFFCURVE:
                return None
            
            # The opposite handle (from next_oncurve side)
            opp_handle_idx = (next_oncurve_idx - 1) % num_nodes
            opp_handle = nodes[opp_handle_idx]
            if opp_handle.type != GSOFFCURVE:
                return None  # Line segment, not curve
            
            p0 = (node.position.x, node.position.y)
            p1 = (our_handle.position.x, our_handle.position.y)
            p2 = (opp_handle.position.x, opp_handle.position.y)
            p3 = (next_oncurve.position.x, next_oncurve.position.y)
            
            return {
                'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3,
                'opp_handle': opp_handle,
                'opp_oncurve_idx': next_oncurve_idx,
                'midpoint': SynchronizationHelper.get_curve_midpoint(p0, p1, p2, p3),
                'anchor': p3,  # p2 scales relative to p3
            }
    
    # -------------------------------------------------------------------------
    # Auto Mode: Optimal Balance Between Dekinking and Curve Preservation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def sample_curve(p0, p1, p2, p3, num_samples=5):
        """Sample a curve at multiple t values for shape comparison."""
        samples = []
        for i in range(num_samples):
            t = (i + 1) / (num_samples + 1)  # Exclude endpoints
            samples.append(SynchronizationHelper.bezier_point(p0, p1, p2, p3, t))
        return samples
    
    @staticmethod
    def curve_shape_deviation(original_samples, new_samples):
        """
        Calculate total deviation between two sets of curve samples.
        Returns the sum of squared distances.
        """
        if len(original_samples) != len(new_samples):
            return float('inf')
        
        total = 0
        for orig, new in zip(original_samples, new_samples):
            dx = new[0] - orig[0]
            dy = new[1] - orig[1]
            total += dx * dx + dy * dy
        return total
    
    @staticmethod
    def calculate_kink_severity(current_ratio, target_ratio):
        """
        Calculate how severe a kink is based on ratio difference.
        Returns a value where 0 means no kink, higher means worse kink.
        """
        if target_ratio <= 0 or current_ratio <= 0:
            return 0
        # Use log ratio difference - this gives symmetric measurement
        log_diff = abs(math.log(current_ratio) - math.log(target_ratio))
        return log_diff
    
    @staticmethod
    def calculate_positions_for_blend(node, path, target_ratio, blend_factor, off_curve_change=0.5):
        """
        Calculate new positions for a given blend factor (0 = no change, 1 = full sync).
        
        Returns dict with:
            - node_pos: New position for the on-curve node
            - prev_pos: New position for prev handle (if off-curve)
            - next_pos: New position for next handle (if off-curve)
            - prev_node: The prev node object
            - next_node: The next node object
        """
        nodes = list(path.nodes)
        num_nodes = len(nodes)
        node_idx = nodes.index(node)
        prev_node = nodes[(node_idx - 1) % num_nodes]
        next_node = nodes[(node_idx + 1) % num_nodes]
        
        # Current positions
        p1 = (prev_node.position.x, prev_node.position.y)
        p2 = (node.position.x, node.position.y)
        p3 = (next_node.position.x, next_node.position.y)
        
        # Calculate full-sync positions
        factor = target_ratio / (target_ratio + 1)
        in_factor = -target_ratio
        out_factor = (target_ratio + 1) / target_ratio
        
        off_curve_count = sum(1 for pt in [prev_node, next_node] if pt.type == GSOFFCURVE)
        if off_curve_count > 0:
            off_soften = off_curve_change / off_curve_count
        else:
            off_soften = 0
        on_soften = 1 - off_soften * off_curve_count
        
        # Full sync positions
        full_pt_x = SynchronizationHelper.lerp(factor, p1[0], p3[0])
        full_pt_y = SynchronizationHelper.lerp(factor, p1[1], p3[1])
        full_pt_x = SynchronizationHelper.lerp(on_soften, p2[0], full_pt_x)
        full_pt_y = SynchronizationHelper.lerp(on_soften, p2[1], full_pt_y)
        
        full_prev_x = SynchronizationHelper.lerp(in_factor, p2[0], p3[0])
        full_prev_y = SynchronizationHelper.lerp(in_factor, p2[1], p3[1])
        full_prev_x = SynchronizationHelper.lerp(off_soften, p1[0], full_prev_x)
        full_prev_y = SynchronizationHelper.lerp(off_soften, p1[1], full_prev_y)
        
        full_next_x = SynchronizationHelper.lerp(out_factor, p1[0], p2[0])
        full_next_y = SynchronizationHelper.lerp(out_factor, p1[1], p2[1])
        full_next_x = SynchronizationHelper.lerp(off_soften, p3[0], full_next_x)
        full_next_y = SynchronizationHelper.lerp(off_soften, p3[1], full_next_y)
        
        # Blend between current and full sync
        new_pt = (
            SynchronizationHelper.lerp(blend_factor, p2[0], full_pt_x),
            SynchronizationHelper.lerp(blend_factor, p2[1], full_pt_y)
        )
        new_prev = (
            SynchronizationHelper.lerp(blend_factor, p1[0], full_prev_x),
            SynchronizationHelper.lerp(blend_factor, p1[1], full_prev_y)
        )
        new_next = (
            SynchronizationHelper.lerp(blend_factor, p3[0], full_next_x),
            SynchronizationHelper.lerp(blend_factor, p3[1], full_next_y)
        )
        
        return {
            'node_pos': new_pt,
            'prev_pos': new_prev,
            'next_pos': new_next,
            'prev_node': prev_node,
            'next_node': next_node,
            'original_prev': p1,
            'original_node': p2,
            'original_next': p3,
        }
    
    @staticmethod
    def find_optimal_blend_for_node(node, path, target_ratio, max_deviation=25.0, steps=21):
        """
        Find the optimal blend factor that maximizes dekinking while keeping
        curve deviation below the threshold.
        
        Args:
            node: The smooth on-curve node
            path: The path containing the node
            target_ratio: The target ratio to sync toward
            max_deviation: Maximum allowed curve deviation (sum of squared distances)
            steps: Number of blend values to test
            
        Returns:
            Optimal blend factor (0.0 to 1.0)
        """
        nodes = list(path.nodes)
        num_nodes = len(nodes)
        node_idx = nodes.index(node)
        prev_node = nodes[(node_idx - 1) % num_nodes]
        next_node = nodes[(node_idx + 1) % num_nodes]
        
        # Sample original curves adjacent to this node
        original_prev_samples = None
        original_next_samples = None
        
        # Get curve info for adjacent curves
        prev_info = SynchronizationHelper.get_curve_segment_info(node, path, "prev")
        next_info = SynchronizationHelper.get_curve_segment_info(node, path, "next")
        
        if prev_info:
            original_prev_samples = SynchronizationHelper.sample_curve(
                prev_info['p0'], prev_info['p1'], prev_info['p2'], prev_info['p3']
            )
        
        if next_info:
            original_next_samples = SynchronizationHelper.sample_curve(
                next_info['p0'], next_info['p1'], next_info['p2'], next_info['p3']
            )
        
        # If no curves to preserve, just do full sync
        if original_prev_samples is None and original_next_samples is None:
            return 1.0
        
        best_blend = 0.0
        current_ratio = SynchronizationHelper.calc_ratio(node, path)
        initial_kink = SynchronizationHelper.calculate_kink_severity(current_ratio, target_ratio)
        
        # If already close to target, no need to change
        if initial_kink < 0.01:
            return 0.0
        
        # Try different blend factors and find the highest one with acceptable deviation
        for i in range(steps):
            blend = i / (steps - 1)
            
            # Calculate positions for this blend
            positions = SynchronizationHelper.calculate_positions_for_blend(
                node, path, target_ratio, blend, off_curve_change=0.5
            )
            
            # Calculate curve deviation for this blend
            total_deviation = 0
            
            if prev_info and original_prev_samples:
                # Rebuild the prev curve with new positions
                # Curve: p0 (fixed) -> p1 (opp_handle, we might compensate) -> p2 (our handle) -> p3 (our node)
                new_p2 = positions['prev_pos'] if prev_node.type == GSOFFCURVE else prev_info['p2']
                new_p3 = positions['node_pos']
                
                # For now, don't compensate opposite handle in deviation calculation
                new_prev_samples = SynchronizationHelper.sample_curve(
                    prev_info['p0'], prev_info['p1'], new_p2, new_p3
                )
                total_deviation += SynchronizationHelper.curve_shape_deviation(
                    original_prev_samples, new_prev_samples
                )
            
            if next_info and original_next_samples:
                # Rebuild the next curve with new positions
                # Curve: p0 (our node) -> p1 (our handle) -> p2 (opp_handle, we might compensate) -> p3 (fixed)
                new_p0 = positions['node_pos']
                new_p1 = positions['next_pos'] if next_node.type == GSOFFCURVE else next_info['p1']
                
                new_next_samples = SynchronizationHelper.sample_curve(
                    new_p0, new_p1, next_info['p2'], next_info['p3']
                )
                total_deviation += SynchronizationHelper.curve_shape_deviation(
                    original_next_samples, new_next_samples
                )
            
            # If deviation is acceptable, this is a candidate
            if total_deviation <= max_deviation:
                best_blend = blend
        
        return best_blend
    
    @staticmethod
    def find_optimal_ratio_across_masters(node, path, path_idx, node_idx, glyph, font, max_deviation=25.0):
        """
        Find the optimal target ratio that best fixes kinks across ALL masters.
        
        Strategy:
        - Use the AVERAGE ratio as the target (best for dekinking)
        - If curves across masters are already similar, be aggressive
        - The per-master blend will handle individual tolerance
        
        Args:
            node: The node in the current layer (for reference)
            path: The path in the current layer
            path_idx: Path index
            node_idx: Node index
            glyph: The GSGlyph
            font: The GSFont
            max_deviation: Base maximum allowed deviation
            
        Returns:
            (target_ratio, adjusted_max_deviation) or (None, None) if no sync needed
        """
        # Collect current ratios and curve samples from all masters
        master_data = []
        
        for master in font.masters:
            master_layer = glyph.layers[master.id]
            if not master_layer:
                continue
            
            corresponding = SynchronizationHelper.get_corresponding_node(
                node, path_idx, node_idx, master_layer
            )
            if not corresponding:
                continue
            
            try:
                master_path = master_layer.paths[path_idx]
                current_ratio = SynchronizationHelper.calc_ratio(corresponding, master_path)
                
                # Get curve info for this master
                prev_info = SynchronizationHelper.get_curve_segment_info(corresponding, master_path, "prev")
                next_info = SynchronizationHelper.get_curve_segment_info(corresponding, master_path, "next")
                
                # Sample original curves
                prev_samples = None
                next_samples = None
                if prev_info:
                    prev_samples = SynchronizationHelper.sample_curve(
                        prev_info['p0'], prev_info['p1'], prev_info['p2'], prev_info['p3']
                    )
                if next_info:
                    next_samples = SynchronizationHelper.sample_curve(
                        next_info['p0'], next_info['p1'], next_info['p2'], next_info['p3']
                    )
                
                master_data.append({
                    'layer': master_layer,
                    'path': master_path,
                    'node': corresponding,
                    'ratio': current_ratio,
                    'prev_info': prev_info,
                    'next_info': next_info,
                    'prev_samples': prev_samples,
                    'next_samples': next_samples,
                })
            except Exception:
                continue
        
        if len(master_data) < 2:
            return None, None
        
        # Get the range of ratios across masters
        ratios = [d['ratio'] for d in master_data]
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        ratio_spread = max_ratio - min_ratio
        
        # If ratios are already very close, no need to change
        if ratio_spread < 0.01:
            return None, None
        
        # TARGET: Use average ratio - this is the best target for dekinking
        target_ratio = sum(ratios) / len(ratios)
        
        # Calculate curve similarity across masters
        # Compare each master's curves to the first master's curves
        curve_similarity = SynchronizationHelper.calculate_cross_master_curve_similarity(master_data)
        
        # Adjust max_deviation based on curve similarity
        # If curves are 95%+ similar across masters, we can be very aggressive
        # similarity of 1.0 = identical curves, allow much more deviation
        # similarity of 0.5 = quite different curves, use base max_deviation
        
        if curve_similarity > 0.95:
            # Curves are nearly identical - be very aggressive, allow 4x deviation
            adjusted_max_deviation = max_deviation * 4.0
        elif curve_similarity > 0.85:
            # Curves are similar - be aggressive, allow 2.5x deviation
            adjusted_max_deviation = max_deviation * 2.5
        elif curve_similarity > 0.70:
            # Curves are somewhat similar - allow 1.5x deviation
            adjusted_max_deviation = max_deviation * 1.5
        else:
            # Curves are quite different - use base deviation
            adjusted_max_deviation = max_deviation
        
        return target_ratio, adjusted_max_deviation
    
    @staticmethod
    def calculate_cross_master_curve_similarity(master_data):
        """
        Calculate how similar the curves are across all masters.
        
        Returns a value from 0 to 1, where:
        - 1.0 = curves are identical across all masters
        - 0.0 = curves are completely different
        
        We compare curve shapes (normalized) rather than absolute positions.
        """
        if len(master_data) < 2:
            return 1.0
        
        # Use first master as reference
        ref_data = master_data[0]
        
        total_similarity = 0
        comparison_count = 0
        
        for other_data in master_data[1:]:
            # Compare prev curves
            if ref_data['prev_samples'] and other_data['prev_samples']:
                sim = SynchronizationHelper.normalized_curve_similarity(
                    ref_data['prev_samples'], other_data['prev_samples']
                )
                total_similarity += sim
                comparison_count += 1
            
            # Compare next curves
            if ref_data['next_samples'] and other_data['next_samples']:
                sim = SynchronizationHelper.normalized_curve_similarity(
                    ref_data['next_samples'], other_data['next_samples']
                )
                total_similarity += sim
                comparison_count += 1
        
        if comparison_count == 0:
            return 1.0
        
        return total_similarity / comparison_count
    
    @staticmethod
    def normalized_curve_similarity(samples1, samples2):
        """
        Calculate similarity between two curves after normalizing their bounding boxes.
        
        This measures shape similarity regardless of size/position differences.
        Returns 0-1 where 1 is identical shape.
        """
        if not samples1 or not samples2 or len(samples1) != len(samples2):
            return 1.0
        
        # Normalize both sample sets to 0-1 range
        def normalize_samples(samples):
            xs = [p[0] for p in samples]
            ys = [p[1] for p in samples]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            range_x = max_x - min_x if max_x > min_x else 1.0
            range_y = max_y - min_y if max_y > min_y else 1.0
            
            # Use the larger range to maintain aspect ratio
            scale = max(range_x, range_y)
            if scale < 0.1:
                scale = 1.0
            
            return [((p[0] - min_x) / scale, (p[1] - min_y) / scale) for p in samples]
        
        norm1 = normalize_samples(samples1)
        norm2 = normalize_samples(samples2)
        
        # Calculate average distance between corresponding normalized points
        total_dist = 0
        for p1, p2 in zip(norm1, norm2):
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            total_dist += (dx * dx + dy * dy) ** 0.5
        
        avg_dist = total_dist / len(norm1)
        
        # Convert distance to similarity (0-1)
        # Distance of 0 = similarity 1, distance of 0.5 = similarity ~0.5
        similarity = max(0, 1.0 - avg_dist * 2)
        
        return similarity
    
    @staticmethod
    def calculate_deviation_for_ratio(node, path, target_ratio, prev_info, next_info, 
                                       prev_samples, next_samples):
        """
        Calculate how much the curves would deviate if we sync to target_ratio.
        """
        nodes = list(path.nodes)
        num_nodes = len(nodes)
        node_idx = nodes.index(node)
        prev_node = nodes[(node_idx - 1) % num_nodes]
        next_node = nodes[(node_idx + 1) % num_nodes]
        
        # Calculate new positions for full sync (blend=1.0)
        positions = SynchronizationHelper.calculate_positions_for_blend(
            node, path, target_ratio, blend_factor=1.0, off_curve_change=0.5
        )
        
        total_deviation = 0
        
        if prev_info and prev_samples and prev_node.type == GSOFFCURVE:
            new_p2 = positions['prev_pos']
            new_p3 = positions['node_pos']
            
            new_samples = SynchronizationHelper.sample_curve(
                prev_info['p0'], prev_info['p1'], new_p2, new_p3
            )
            total_deviation += SynchronizationHelper.curve_shape_deviation(prev_samples, new_samples)
        
        if next_info and next_samples and next_node.type == GSOFFCURVE:
            new_p0 = positions['node_pos']
            new_p1 = positions['next_pos']
            
            new_samples = SynchronizationHelper.sample_curve(
                new_p0, new_p1, next_info['p2'], next_info['p3']
            )
            total_deviation += SynchronizationHelper.curve_shape_deviation(next_samples, new_samples)
        
        return total_deviation
    
    @staticmethod
    def sync_ratios_auto(selected_nodes_info, font, max_deviation=40.0):
        """
        Automatically dekink while preserving curve shape as much as possible.
        
        Strategy:
        - Find optimal target ratio (average across masters)
        - If curves are similar across masters, be aggressive about syncing
        - Apply compensation to preserve curve midpoints
        
        Args:
            selected_nodes_info: List of (node, path, path_idx, node_idx)
            font: The GSFont
            max_deviation: Base maximum allowed curve deviation
            
        Returns:
            Number of nodes processed
        """
        if not selected_nodes_info or not font:
            return 0
        
        current_layer = font.selectedLayers[0] if font.selectedLayers else None
        if not current_layer:
            return 0
        
        glyph = current_layer.parent
        if not glyph:
            return 0
        
        processed_count = 0
        
        # Process each selected node
        for node, path, path_idx, node_idx in selected_nodes_info:
            # Find optimal ratio across ALL masters (and adjusted deviation threshold)
            optimal_ratio, adjusted_max_dev = SynchronizationHelper.find_optimal_ratio_across_masters(
                node, path, path_idx, node_idx, glyph, font, max_deviation
            )
            
            if optimal_ratio is None:
                # No sync needed or not enough masters
                continue
            
            # For each master, apply the sync with the adjusted threshold
            for master in font.masters:
                master_layer = glyph.layers[master.id]
                if not master_layer:
                    continue
                
                corresponding = SynchronizationHelper.get_corresponding_node(
                    node, path_idx, node_idx, master_layer
                )
                if not corresponding:
                    continue
                
                try:
                    master_path = master_layer.paths[path_idx]
                    
                    # Find optimal blend for this master toward the global optimal ratio
                    # Use the ADJUSTED max deviation (higher if curves are similar)
                    optimal_blend = SynchronizationHelper.find_optimal_blend_for_node(
                        corresponding, master_path, optimal_ratio, adjusted_max_dev
                    )
                    
                    if optimal_blend < 0.01:
                        continue
                    
                    # Capture curve midpoints before applying changes (for compensation)
                    prev_info = SynchronizationHelper.get_curve_segment_info(
                        corresponding, master_path, "prev"
                    )
                    next_info = SynchronizationHelper.get_curve_segment_info(
                        corresponding, master_path, "next"
                    )
                    
                    # Get positions for this blend
                    positions = SynchronizationHelper.calculate_positions_for_blend(
                        corresponding, master_path, optimal_ratio, optimal_blend, off_curve_change=0.5
                    )
                    
                    # Apply the positions to the selected node and its DIRECT handles only
                    corresponding.position = NSPoint(
                        round(positions['node_pos'][0]), 
                        round(positions['node_pos'][1])
                    )
                    
                    prev_node = positions['prev_node']
                    next_node = positions['next_node']
                    
                    if prev_node.type == GSOFFCURVE:
                        prev_node.position = NSPoint(
                            round(positions['prev_pos'][0]),
                            round(positions['prev_pos'][1])
                        )
                    
                    if next_node.type == GSOFFCURVE:
                        next_node.position = NSPoint(
                            round(positions['next_pos'][0]),
                            round(positions['next_pos'][1])
                        )
                    
                    # Apply curve compensation to opposite handles
                    if prev_info and prev_node.type == GSOFFCURVE:
                        opp_handle = prev_info['opp_handle']
                        target_mid = prev_info['midpoint']
                        anchor = prev_info['anchor']
                        
                        p0 = anchor
                        p2_new = (prev_node.position.x, prev_node.position.y)
                        p3_new = (corresponding.position.x, corresponding.position.y)
                        
                        p1_current = (opp_handle.position.x, opp_handle.position.y)
                        p1_dir = (p1_current[0] - p0[0], p1_current[1] - p0[1])
                        
                        if abs(p1_dir[0]) > 0.1 or abs(p1_dir[1]) > 0.1:
                            best_scale = SynchronizationHelper.find_handle_scale_for_midpoint(
                                p0, p1_dir, p2_new, p3_new, target_mid
                            )
                            new_p1 = (p0[0] + p1_dir[0] * best_scale, 
                                     p0[1] + p1_dir[1] * best_scale)
                            opp_handle.position = NSPoint(round(new_p1[0]), round(new_p1[1]))
                    
                    if next_info and next_node.type == GSOFFCURVE:
                        opp_handle = next_info['opp_handle']
                        target_mid = next_info['midpoint']
                        anchor = next_info['anchor']
                        
                        p0_new = (corresponding.position.x, corresponding.position.y)
                        p1_new = (next_node.position.x, next_node.position.y)
                        p3 = anchor
                        
                        p2_current = (opp_handle.position.x, opp_handle.position.y)
                        p2_dir = (p2_current[0] - p3[0], p2_current[1] - p3[1])
                        
                        if abs(p2_dir[0]) > 0.1 or abs(p2_dir[1]) > 0.1:
                            best_scale = SynchronizationHelper.find_handle_scale_for_midpoint_p2(
                                p0_new, p1_new, p2_dir, p3, target_mid
                            )
                            new_p2 = (p3[0] + p2_dir[0] * best_scale,
                                     p3[1] + p2_dir[1] * best_scale)
                            opp_handle.position = NSPoint(round(new_p2[0]), round(new_p2[1]))
                
                except Exception:
                    continue
            
            processed_count += 1
        
        return processed_count
    
    @staticmethod
    def calculate_preview_points_auto(selected_nodes_info, font, max_deviation=40.0):
        """
        Calculate preview points for auto mode.
        """
        if not selected_nodes_info or not font:
            return None
        
        current_layer = font.selectedLayers[0] if font.selectedLayers else None
        if not current_layer:
            return None
        
        glyph = current_layer.parent
        if not glyph:
            return None
        
        # Extract current glyph points for all masters
        preview_points = {}
        master_layers = []
        
        for ix, master in enumerate(font.masters):
            layer = glyph.layers[master.id]
            if not layer:
                return None
            master_layers.append(layer)
            decomposed = GlyphInterpolator.get_decomposed_layer(layer)
            preview_points[ix] = list(GlyphInterpolator.extract_points_from_layer(decomposed))
        
        # Build node-to-point mapping
        first_decomposed = GlyphInterpolator.get_decomposed_layer(master_layers[0])
        node_to_point = GlyphInterpolator.build_node_to_point_map(first_decomposed)
        
        # Process each selected node
        for node, path, path_idx, node_idx in selected_nodes_info:
            # Find optimal ratio across ALL masters
            optimal_ratio, adjusted_max_dev = SynchronizationHelper.find_optimal_ratio_across_masters(
                node, path, path_idx, node_idx, glyph, font, max_deviation
            )
            
            if optimal_ratio is None:
                continue
            
            for master_idx, master in enumerate(font.masters):
                master_layer = master_layers[master_idx]
                
                corresponding = SynchronizationHelper.get_corresponding_node(
                    node, path_idx, node_idx, master_layer
                )
                if not corresponding:
                    continue
                
                try:
                    master_path = master_layer.paths[path_idx]
                    nodes = list(master_path.nodes)
                    num_nodes = len(nodes)
                    corr_node_idx = nodes.index(corresponding)
                    
                    # Find optimal blend toward the global optimal ratio
                    optimal_blend = SynchronizationHelper.find_optimal_blend_for_node(
                        corresponding, master_path, optimal_ratio, adjusted_max_dev
                    )
                    
                    if optimal_blend < 0.01:
                        continue
                    
                    # Get positions for this blend
                    positions = SynchronizationHelper.calculate_positions_for_blend(
                        corresponding, master_path, optimal_ratio, optimal_blend, off_curve_change=0.5
                    )
                    
                    prev_node = positions['prev_node']
                    next_node = positions['next_node']
                    
                    # Find flat indices
                    node_flat_idx = node_to_point.get((path_idx, node_idx))
                    prev_node_idx_in_path = (corr_node_idx - 1) % num_nodes
                    next_node_idx_in_path = (corr_node_idx + 1) % num_nodes
                    prev_flat_idx = node_to_point.get((path_idx, prev_node_idx_in_path))
                    next_flat_idx = node_to_point.get((path_idx, next_node_idx_in_path))
                    
                    # Update preview points for the selected node and its direct handles
                    if node_flat_idx is not None and node_flat_idx < len(preview_points[master_idx]):
                        preview_points[master_idx][node_flat_idx] = (
                            round(positions['node_pos'][0]), 
                            round(positions['node_pos'][1])
                        )
                    
                    if prev_node.type == GSOFFCURVE and prev_flat_idx is not None:
                        if prev_flat_idx < len(preview_points[master_idx]):
                            preview_points[master_idx][prev_flat_idx] = (
                                round(positions['prev_pos'][0]),
                                round(positions['prev_pos'][1])
                            )
                    
                    if next_node.type == GSOFFCURVE and next_flat_idx is not None:
                        if next_flat_idx < len(preview_points[master_idx]):
                            preview_points[master_idx][next_flat_idx] = (
                                round(positions['next_pos'][0]),
                                round(positions['next_pos'][1])
                            )
                    
                    # Also show compensation on opposite handles in preview
                    prev_info = SynchronizationHelper.get_curve_segment_info(
                        corresponding, master_path, "prev"
                    )
                    next_info = SynchronizationHelper.get_curve_segment_info(
                        corresponding, master_path, "next"
                    )
                    
                    if prev_info and prev_node.type == GSOFFCURVE:
                        opp_handle = prev_info['opp_handle']
                        target_mid = prev_info['midpoint']
                        anchor = prev_info['anchor']
                        
                        p0 = anchor
                        p2_new = positions['prev_pos']
                        p3_new = positions['node_pos']
                        
                        p1_current = (opp_handle.position.x, opp_handle.position.y)
                        p1_dir = (p1_current[0] - p0[0], p1_current[1] - p0[1])
                        
                        if abs(p1_dir[0]) > 0.1 or abs(p1_dir[1]) > 0.1:
                            best_scale = SynchronizationHelper.find_handle_scale_for_midpoint(
                                p0, p1_dir, p2_new, p3_new, target_mid
                            )
                            new_p1 = (p0[0] + p1_dir[0] * best_scale, 
                                     p0[1] + p1_dir[1] * best_scale)
                            
                            opp_handle_idx = nodes.index(opp_handle)
                            opp_flat_idx = node_to_point.get((path_idx, opp_handle_idx))
                            if opp_flat_idx is not None and opp_flat_idx < len(preview_points[master_idx]):
                                preview_points[master_idx][opp_flat_idx] = (round(new_p1[0]), round(new_p1[1]))
                    
                    if next_info and next_node.type == GSOFFCURVE:
                        opp_handle = next_info['opp_handle']
                        target_mid = next_info['midpoint']
                        anchor = next_info['anchor']
                        
                        p0_new = positions['node_pos']
                        p1_new = positions['next_pos']
                        p3 = anchor
                        
                        p2_current = (opp_handle.position.x, opp_handle.position.y)
                        p2_dir = (p2_current[0] - p3[0], p2_current[1] - p3[1])
                        
                        if abs(p2_dir[0]) > 0.1 or abs(p2_dir[1]) > 0.1:
                            best_scale = SynchronizationHelper.find_handle_scale_for_midpoint_p2(
                                p0_new, p1_new, p2_dir, p3, target_mid
                            )
                            new_p2 = (p3[0] + p2_dir[0] * best_scale,
                                     p3[1] + p2_dir[1] * best_scale)
                            
                            opp_handle_idx = nodes.index(opp_handle)
                            opp_flat_idx = node_to_point.get((path_idx, opp_handle_idx))
                            if opp_flat_idx is not None and opp_flat_idx < len(preview_points[master_idx]):
                                preview_points[master_idx][opp_flat_idx] = (round(new_p2[0]), round(new_p2[1]))
                
                except Exception:
                    continue
        
        # Convert to GlyphCoordinates
        result = {}
        for master_idx, points_list in preview_points.items():
            result[master_idx] = GlyphCoordinates(points_list)
        
        return result
    
    # -------------------------------------------------------------------------
    # Core Ratio Methods
    # -------------------------------------------------------------------------
    
    @staticmethod
    def get_surrounding_points(node, path):
        """
        Get the previous and next nodes for a given node in a path.
        
        Args:
            node: The GSNode to find neighbors for
            path: The GSPath containing the node
            
        Returns:
            Tuple of (prev_node, next_node)
        """
        nodes = list(path.nodes)
        num_nodes = len(nodes)
        node_idx = nodes.index(node)
        prev_node = nodes[(node_idx - 1) % num_nodes]
        next_node = nodes[(node_idx + 1) % num_nodes]
        return prev_node, next_node
    
    @staticmethod
    def calc_ratio(node, path):
        """
        Calculate the distance ratio for a node.
        
        Args:
            node: The GSNode to calculate ratio for
            path: The GSPath containing the node
            
        Returns:
            Float ratio (distance_in / distance_out)
        """
        prev_node, next_node = SynchronizationHelper.get_surrounding_points(node, path)
        
        p1 = (prev_node.position.x, prev_node.position.y)
        p2 = (node.position.x, node.position.y)
        p3 = (next_node.position.x, next_node.position.y)
        
        dist_in = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        dist_out = math.hypot(p3[0] - p2[0], p3[1] - p2[1])
        
        if dist_out == 0:
            return dist_in if dist_in > 0 else 1.0
        return dist_in / dist_out
    
    @staticmethod
    def lerp(factor, v1, v2):
        """Linear interpolation between two values."""
        return v1 + factor * (v2 - v1)
    
    @staticmethod
    def change_ratio(node, path, new_ratio, off_curve_change=0.15, round_coords=True):
        """
        Change the position of a node and optionally its neighbors to achieve a new ratio.
        
        Args:
            node: The GSNode to adjust
            path: The GSPath containing the node
            new_ratio: The target ratio to achieve
            off_curve_change: How much off-curve points should share the adjustment (0-1)
                              0 = only move on-curve, 1 = only move off-curves
            round_coords: Whether to round coordinates to integers
            
        Returns:
            Dict of {node: (old_x, old_y)} for undo purposes
        """
        prev_node, next_node = SynchronizationHelper.get_surrounding_points(node, path)
        
        p1 = (prev_node.position.x, prev_node.position.y)
        p2 = (node.position.x, node.position.y)
        p3 = (next_node.position.x, next_node.position.y)
        
        # Store old positions for undo
        old_positions = {
            node: (node.position.x, node.position.y),
            prev_node: (prev_node.position.x, prev_node.position.y),
            next_node: (next_node.position.x, next_node.position.y)
        }
        
        # Calculate factors for the new position
        factor = new_ratio / (new_ratio + 1)
        in_factor = -new_ratio
        out_factor = (new_ratio + 1) / new_ratio
        
        # Count off-curve neighbors
        off_curve_count = sum(1 for pt in [prev_node, next_node] if pt.type == GSOFFCURVE)
        
        if off_curve_count > 0:
            off_soften = off_curve_change / off_curve_count
        else:
            off_soften = 0
        on_soften = 1 - off_soften * off_curve_count
        
        # Calculate new on-curve position
        new_pt_x = SynchronizationHelper.lerp(factor, p1[0], p3[0])
        new_pt_y = SynchronizationHelper.lerp(factor, p1[1], p3[1])
        new_pt_x = SynchronizationHelper.lerp(on_soften, p2[0], new_pt_x)
        new_pt_y = SynchronizationHelper.lerp(on_soften, p2[1], new_pt_y)
        
        # Calculate new previous point position (if off-curve)
        new_prev_x = SynchronizationHelper.lerp(in_factor, p2[0], p3[0])
        new_prev_y = SynchronizationHelper.lerp(in_factor, p2[1], p3[1])
        new_prev_x = SynchronizationHelper.lerp(off_soften, p1[0], new_prev_x)
        new_prev_y = SynchronizationHelper.lerp(off_soften, p1[1], new_prev_y)
        
        # Calculate new next point position (if off-curve)
        new_next_x = SynchronizationHelper.lerp(out_factor, p1[0], p2[0])
        new_next_y = SynchronizationHelper.lerp(out_factor, p1[1], p2[1])
        new_next_x = SynchronizationHelper.lerp(off_soften, p3[0], new_next_x)
        new_next_y = SynchronizationHelper.lerp(off_soften, p3[1], new_next_y)
        
        # Apply rounding if requested
        if round_coords:
            new_pt_x, new_pt_y = round(new_pt_x), round(new_pt_y)
            new_prev_x, new_prev_y = round(new_prev_x), round(new_prev_y)
            new_next_x, new_next_y = round(new_next_x), round(new_next_y)
        
        # Apply changes
        node.position = NSPoint(new_pt_x, new_pt_y)
        
        if prev_node.type == GSOFFCURVE:
            prev_node.position = NSPoint(new_prev_x, new_prev_y)
        
        if next_node.type == GSOFFCURVE:
            next_node.position = NSPoint(new_next_x, new_next_y)
        
        return old_positions
    
    @staticmethod
    def get_corresponding_node(node, path_index, node_index, target_layer):
        """
        Get the corresponding node in another layer.
        
        Args:
            node: The reference node
            path_index: Index of the path in the layer
            node_index: Index of the node in the path
            target_layer: The layer to find the corresponding node in
            
        Returns:
            The corresponding GSNode, or None if not found
        """
        try:
            if path_index >= len(target_layer.paths):
                return None
            target_path = target_layer.paths[path_index]
            if node_index >= len(target_path.nodes):
                return None
            target_node = target_path.nodes[node_index]
            
            # Verify node type matches
            if target_node.type != node.type:
                return None
            if target_node.smooth != node.smooth:
                return None
            
            return target_node
        except Exception:
            return None
    
    @staticmethod
    def get_selected_smooth_oncurve_nodes(layer):
        """
        Get all selected smooth on-curve nodes from a layer.
        
        Args:
            layer: The GSLayer to check
            
        Returns:
            List of tuples: (node, path, path_index, node_index)
        """
        result = []
        if not layer or not layer.selection:
            return result
        
        for path_idx, path in enumerate(layer.paths):
            nodes = list(path.nodes)
            for node_idx, node in enumerate(nodes):
                if node in layer.selection:
                    # Check if it's a smooth on-curve node
                    if node.type != GSOFFCURVE and node.smooth:
                        result.append((node, path, path_idx, node_idx))
        
        return result
    
    @staticmethod
    def filter_nodes_with_kinks(selected_nodes_info, layer, font):
        """
        Filter selected nodes to only those that have kinks or are curve-segment
        counterparts of nodes that have kinks.
        
        A node is included if:
        1. It has a kink (detected by KinkDetector), OR
        2. It shares a curve segment with a kinked node that is also selected
           (both smooth on-curve points on the same bezier curve)
        
        Args:
            selected_nodes_info: List of (node, path, path_idx, node_idx) from get_selected_smooth_oncurve_nodes
            layer: The current GSLayer
            font: The GSFont
            
        Returns:
            Filtered list of (node, path, path_idx, node_idx) tuples
        """
        if not selected_nodes_info or not layer or not font:
            return selected_nodes_info  # Return as-is if can't filter
        
        # Get the set of nodes that have kinks
        potential_kinks = KinkDetector.find_potential_kinks(layer, font)
        
        # Build a set of (path_idx, node_idx) for kinked nodes
        kinked_nodes = set()
        for kink in potential_kinks:
            # kink format: (x, y, max_angle_diff, node_id, has_ignored_axes)
            # node_id format: "p{path_index}_n{node_index}"
            node_id = kink[3]
            try:
                parts = node_id.split('_')
                p_idx = int(parts[0][1:])  # Remove 'p' prefix
                n_idx = int(parts[1][1:])  # Remove 'n' prefix
                kinked_nodes.add((p_idx, n_idx))
            except (IndexError, ValueError):
                continue
        
        if not kinked_nodes:
            # No kinks detected - return empty list (nothing to sync)
            return []
        
        # Build a set of selected (path_idx, node_idx)
        selected_set = {(p_idx, n_idx) for _, _, p_idx, n_idx in selected_nodes_info}
        
        # For each selected node, determine if it should be included
        result = []
        included_set = set()  # Track what we've already included
        
        for node, path, path_idx, node_idx in selected_nodes_info:
            # Already included?
            if (path_idx, node_idx) in included_set:
                continue
            
            # Check if this node has a kink
            if (path_idx, node_idx) in kinked_nodes:
                result.append((node, path, path_idx, node_idx))
                included_set.add((path_idx, node_idx))
                continue
            
            # Check if this node is a curve-segment counterpart of a kinked node
            # that is also selected. A curve segment connects two on-curve points.
            # We need to find the adjacent on-curve nodes and check if any of them
            # are both kinked AND selected.
            nodes_list = list(path.nodes)
            num_nodes = len(nodes_list)
            
            # Find the previous on-curve node
            prev_oncurve_idx = None
            for i in range(1, num_nodes):
                check_idx = (node_idx - i) % num_nodes
                check_node = nodes_list[check_idx]
                if check_node.type != GSOFFCURVE:
                    prev_oncurve_idx = check_idx
                    break
            
            # Find the next on-curve node
            next_oncurve_idx = None
            for i in range(1, num_nodes):
                check_idx = (node_idx + i) % num_nodes
                check_node = nodes_list[check_idx]
                if check_node.type != GSOFFCURVE:
                    next_oncurve_idx = check_idx
                    break
            
            # Check if either adjacent on-curve is kinked AND selected
            include_as_counterpart = False
            
            if prev_oncurve_idx is not None:
                prev_key = (path_idx, prev_oncurve_idx)
                if prev_key in kinked_nodes and prev_key in selected_set:
                    include_as_counterpart = True
            
            if next_oncurve_idx is not None and not include_as_counterpart:
                next_key = (path_idx, next_oncurve_idx)
                if next_key in kinked_nodes and next_key in selected_set:
                    include_as_counterpart = True
            
            if include_as_counterpart:
                result.append((node, path, path_idx, node_idx))
                included_set.add((path_idx, node_idx))
        
        return result
    
    @staticmethod
    def sync_ratios(selected_nodes_info, font, mode="follow", off_curve_change=0.15, compensate_curves=False):
        """
        Synchronize ratios for selected nodes across all masters.
        
        Args:
            selected_nodes_info: List of (node, path, path_idx, node_idx) from get_selected_smooth_oncurve_nodes
            font: The GSFont
            mode: "follow" to use selected layer's ratios, "average" to average all masters
            off_curve_change: How much off-curves share the adjustment (0-1)
            compensate_curves: If True, adjust opposite handles to preserve curve midpoint
            
        Returns:
            Number of nodes synchronized
        """
        if not selected_nodes_info or not font:
            return 0
        
        synced_count = 0
        current_layer = font.selectedLayers[0] if font.selectedLayers else None
        if not current_layer:
            return 0
        
        glyph = current_layer.parent
        if not glyph:
            return 0
        
        # Build set of selected node indices per path to avoid compensation conflicts
        selected_indices_by_path = {}
        for _, _, p_idx, n_idx in selected_nodes_info:
            if p_idx not in selected_indices_by_path:
                selected_indices_by_path[p_idx] = set()
            selected_indices_by_path[p_idx].add(n_idx)
        
        # Process each selected node
        for node, path, path_idx, node_idx in selected_nodes_info:
            # Calculate target ratio based on mode
            if mode == "follow":
                target_ratio = SynchronizationHelper.calc_ratio(node, path)
            elif mode == "average":
                # Calculate average ratio across all masters
                ratios = []
                for master in font.masters:
                    master_layer = glyph.layers[master.id]
                    if not master_layer:
                        continue
                    corresponding = SynchronizationHelper.get_corresponding_node(
                        node, path_idx, node_idx, master_layer
                    )
                    if corresponding:
                        try:
                            master_path = master_layer.paths[path_idx]
                            ratio = SynchronizationHelper.calc_ratio(corresponding, master_path)
                            ratios.append(ratio)
                        except Exception:
                            continue
                
                if not ratios:
                    continue
                target_ratio = sum(ratios) / len(ratios)
            else:
                continue
            
            # Apply the target ratio to all masters
            for master in font.masters:
                master_layer = glyph.layers[master.id]
                if not master_layer:
                    continue
                
                corresponding = SynchronizationHelper.get_corresponding_node(
                    node, path_idx, node_idx, master_layer
                )
                if not corresponding:
                    continue
                
                try:
                    master_path = master_layer.paths[path_idx]
                    
                    # If compensate_curves is enabled, capture curve midpoints before sync
                    curve_info = {}
                    if compensate_curves:
                        selected_in_path = selected_indices_by_path.get(path_idx, set())
                        
                        # Capture "prev" curve info
                        prev_info = SynchronizationHelper.get_curve_segment_info(
                            corresponding, master_path, "prev"
                        )
                        if prev_info and prev_info['opp_oncurve_idx'] not in selected_in_path:
                            curve_info['prev'] = prev_info
                        
                        # Capture "next" curve info  
                        next_info = SynchronizationHelper.get_curve_segment_info(
                            corresponding, master_path, "next"
                        )
                        if next_info and next_info['opp_oncurve_idx'] not in selected_in_path:
                            curve_info['next'] = next_info
                    
                    # Apply the ratio change (this moves our node's handles)
                    SynchronizationHelper.change_ratio(
                        corresponding, master_path, target_ratio, off_curve_change
                    )
                    
                    # Now compensate opposite handles to preserve curve midpoints
                    if compensate_curves:
                        prev_node, next_node = SynchronizationHelper.get_surrounding_points(
                            corresponding, master_path
                        )
                        
                        # Compensate "prev" curve
                        # Structure: p0 (prev_oncurve) -> p1 (opp_handle) -> p2 (our handle) -> p3 (corresponding)
                        # p2 has moved, we need to adjust p1
                        if 'prev' in curve_info and prev_node.type == GSOFFCURVE:
                            info = curve_info['prev']
                            opp_handle = info['opp_handle']
                            target_mid = info['midpoint']
                            anchor = info['anchor']  # p0
                            
                            # Get new positions after ratio change
                            p0 = anchor
                            p2_new = (prev_node.position.x, prev_node.position.y)
                            p3_new = (corresponding.position.x, corresponding.position.y)
                            
                            # Direction from p0 to current p1
                            p1_current = (opp_handle.position.x, opp_handle.position.y)
                            p1_dir = (p1_current[0] - p0[0], p1_current[1] - p0[1])
                            
                            # Only compensate if we have a meaningful handle
                            if abs(p1_dir[0]) > 0.1 or abs(p1_dir[1]) > 0.1:
                                # Find scale that preserves midpoint
                                best_scale = SynchronizationHelper.find_handle_scale_for_midpoint(
                                    p0, p1_dir, p2_new, p3_new, target_mid
                                )
                                
                                new_p1 = (p0[0] + p1_dir[0] * best_scale, 
                                         p0[1] + p1_dir[1] * best_scale)
                                opp_handle.position = NSPoint(round(new_p1[0]), round(new_p1[1]))
                        
                        # Compensate "next" curve
                        # Structure: p0 (corresponding) -> p1 (our handle) -> p2 (opp_handle) -> p3 (next_oncurve)
                        # p1 has moved, we need to adjust p2
                        if 'next' in curve_info and next_node.type == GSOFFCURVE:
                            info = curve_info['next']
                            opp_handle = info['opp_handle']
                            target_mid = info['midpoint']
                            anchor = info['anchor']  # p3
                            
                            # Get new positions after ratio change
                            p0_new = (corresponding.position.x, corresponding.position.y)
                            p1_new = (next_node.position.x, next_node.position.y)
                            p3 = anchor
                            
                            # Direction from p3 to current p2
                            p2_current = (opp_handle.position.x, opp_handle.position.y)
                            p2_dir = (p2_current[0] - p3[0], p2_current[1] - p3[1])
                            
                            # Only compensate if we have a meaningful handle
                            if abs(p2_dir[0]) > 0.1 or abs(p2_dir[1]) > 0.1:
                                # Find scale that preserves midpoint
                                best_scale = SynchronizationHelper.find_handle_scale_for_midpoint_p2(
                                    p0_new, p1_new, p2_dir, p3, target_mid
                                )
                                
                                new_p2 = (p3[0] + p2_dir[0] * best_scale,
                                         p3[1] + p2_dir[1] * best_scale)
                                opp_handle.position = NSPoint(round(new_p2[0]), round(new_p2[1]))
                    
                except Exception:
                    continue
            
            synced_count += 1
        
        return synced_count
    
    @staticmethod
    def calculate_preview_points(selected_nodes_info, font, mode="follow", off_curve_change=0.15, compensate_curves=False):
        """
        Calculate what the glyph points would look like after synchronization,
        WITHOUT actually modifying the glyph. Used for live preview.
        
        Args:
            selected_nodes_info: List of (node, path, path_idx, node_idx) from get_selected_smooth_oncurve_nodes
            font: The GSFont
            mode: "follow" to use selected layer's ratios, "average" to average all masters
            off_curve_change: How much off-curves share the adjustment (0-1)
            compensate_curves: If True, adjust opposite handles to preserve curve midpoint
            
        Returns:
            Dict mapping master_index -> GlyphCoordinates with the preview points,
            or None if preview cannot be calculated
        """
        if not selected_nodes_info or not font:
            return None
        
        current_layer = font.selectedLayers[0] if font.selectedLayers else None
        if not current_layer:
            return None
        
        glyph = current_layer.parent
        if not glyph:
            return None
        
        # First, extract the current glyph points for all masters
        # We'll create modified copies
        preview_points = {}
        master_layers = []
        
        for ix, master in enumerate(font.masters):
            layer = glyph.layers[master.id]
            if not layer:
                return None
            master_layers.append(layer)
            decomposed = GlyphInterpolator.get_decomposed_layer(layer)
            preview_points[ix] = list(GlyphInterpolator.extract_points_from_layer(decomposed))
        
        # Build node-to-point mapping using first master's decomposed layer
        first_decomposed = GlyphInterpolator.get_decomposed_layer(master_layers[0])
        node_to_point = GlyphInterpolator.build_node_to_point_map(first_decomposed)
        
        # Build set of selected node indices per path to avoid compensation conflicts
        selected_indices_by_path = {}
        for _, _, p_idx, n_idx in selected_nodes_info:
            if p_idx not in selected_indices_by_path:
                selected_indices_by_path[p_idx] = set()
            selected_indices_by_path[p_idx].add(n_idx)
        
        # Process each selected node
        for node, path, path_idx, node_idx in selected_nodes_info:
            # Calculate target ratio based on mode
            if mode == "follow":
                target_ratio = SynchronizationHelper.calc_ratio(node, path)
            elif mode == "average":
                # Calculate average ratio across all masters
                ratios = []
                for master_idx, master in enumerate(font.masters):
                    master_layer = master_layers[master_idx]
                    corresponding = SynchronizationHelper.get_corresponding_node(
                        node, path_idx, node_idx, master_layer
                    )
                    if corresponding:
                        try:
                            master_path = master_layer.paths[path_idx]
                            ratio = SynchronizationHelper.calc_ratio(corresponding, master_path)
                            ratios.append(ratio)
                        except Exception:
                            continue
                
                if not ratios:
                    continue
                target_ratio = sum(ratios) / len(ratios)
            else:
                continue
            
            # Calculate the new positions for each master and update preview_points
            for master_idx, master in enumerate(font.masters):
                master_layer = master_layers[master_idx]
                
                corresponding = SynchronizationHelper.get_corresponding_node(
                    node, path_idx, node_idx, master_layer
                )
                if not corresponding:
                    continue
                
                try:
                    master_path = master_layer.paths[path_idx]
                    
                    # Get the surrounding points
                    nodes = list(master_path.nodes)
                    num_nodes = len(nodes)
                    corr_node_idx = nodes.index(corresponding)
                    prev_node = nodes[(corr_node_idx - 1) % num_nodes]
                    next_node = nodes[(corr_node_idx + 1) % num_nodes]
                    
                    p1 = (prev_node.position.x, prev_node.position.y)
                    p2 = (corresponding.position.x, corresponding.position.y)
                    p3 = (next_node.position.x, next_node.position.y)
                    
                    # If compensate_curves is enabled, capture curve midpoints BEFORE calculating new positions
                    curve_compensation = {}
                    if compensate_curves:
                        selected_in_path = selected_indices_by_path.get(path_idx, set())
                        
                        # Capture "prev" curve info (curve coming INTO this node)
                        prev_info = SynchronizationHelper.get_curve_segment_info(
                            corresponding, master_path, "prev"
                        )
                        if prev_info and prev_info['opp_oncurve_idx'] not in selected_in_path:
                            curve_compensation['prev'] = prev_info
                        
                        # Capture "next" curve info (curve going OUT from this node)
                        next_info = SynchronizationHelper.get_curve_segment_info(
                            corresponding, master_path, "next"
                        )
                        if next_info and next_info['opp_oncurve_idx'] not in selected_in_path:
                            curve_compensation['next'] = next_info
                    
                    # Calculate factors for the new position
                    factor = target_ratio / (target_ratio + 1)
                    in_factor = -target_ratio
                    out_factor = (target_ratio + 1) / target_ratio
                    
                    # Count off-curve neighbors
                    off_curve_count = sum(1 for pt in [prev_node, next_node] if pt.type == GSOFFCURVE)
                    
                    if off_curve_count > 0:
                        off_soften = off_curve_change / off_curve_count
                    else:
                        off_soften = 0
                    on_soften = 1 - off_soften * off_curve_count
                    
                    # Calculate new positions for the selected node and its handles
                    new_pt_x = SynchronizationHelper.lerp(factor, p1[0], p3[0])
                    new_pt_y = SynchronizationHelper.lerp(factor, p1[1], p3[1])
                    new_pt_x = SynchronizationHelper.lerp(on_soften, p2[0], new_pt_x)
                    new_pt_y = SynchronizationHelper.lerp(on_soften, p2[1], new_pt_y)
                    
                    new_prev_x = SynchronizationHelper.lerp(in_factor, p2[0], p3[0])
                    new_prev_y = SynchronizationHelper.lerp(in_factor, p2[1], p3[1])
                    new_prev_x = SynchronizationHelper.lerp(off_soften, p1[0], new_prev_x)
                    new_prev_y = SynchronizationHelper.lerp(off_soften, p1[1], new_prev_y)
                    
                    new_next_x = SynchronizationHelper.lerp(out_factor, p1[0], p2[0])
                    new_next_y = SynchronizationHelper.lerp(out_factor, p1[1], p2[1])
                    new_next_x = SynchronizationHelper.lerp(off_soften, p3[0], new_next_x)
                    new_next_y = SynchronizationHelper.lerp(off_soften, p3[1], new_next_y)
                    
                    # Find flat indices for these nodes
                    node_flat_idx = node_to_point.get((path_idx, node_idx))
                    
                    # For prev and next, we need their indices in the path
                    prev_node_idx_in_path = (corr_node_idx - 1) % num_nodes
                    next_node_idx_in_path = (corr_node_idx + 1) % num_nodes
                    
                    prev_flat_idx = node_to_point.get((path_idx, prev_node_idx_in_path))
                    next_flat_idx = node_to_point.get((path_idx, next_node_idx_in_path))
                    
                    # Update the preview points for the selected node and its handles
                    if node_flat_idx is not None and node_flat_idx < len(preview_points[master_idx]):
                        preview_points[master_idx][node_flat_idx] = (round(new_pt_x), round(new_pt_y))
                    
                    if prev_node.type == GSOFFCURVE and prev_flat_idx is not None and prev_flat_idx < len(preview_points[master_idx]):
                        preview_points[master_idx][prev_flat_idx] = (round(new_prev_x), round(new_prev_y))
                    
                    if next_node.type == GSOFFCURVE and next_flat_idx is not None and next_flat_idx < len(preview_points[master_idx]):
                        preview_points[master_idx][next_flat_idx] = (round(new_next_x), round(new_next_y))
                    
                    # Now apply curve compensation if enabled
                    if compensate_curves:
                        # Compensate "prev" curve: p2 (our prev handle) moved, adjust p1 (opposite handle)
                        # Curve: p0 (prev_oncurve) -> p1 (opp_handle) -> p2 (our handle) -> p3 (our node)
                        if 'prev' in curve_compensation and prev_node.type == GSOFFCURVE:
                            info = curve_compensation['prev']
                            opp_handle = info['opp_handle']
                            target_mid = info['midpoint']
                            anchor = info['anchor']  # p0, the previous on-curve
                            
                            # New positions after ratio sync
                            p0 = anchor
                            p2_new = (new_prev_x, new_prev_y)  # Our handle's new position
                            p3_new = (new_pt_x, new_pt_y)  # Our node's new position
                            
                            # Direction from anchor (p0) to current opp_handle (p1)
                            p1_current = (opp_handle.position.x, opp_handle.position.y)
                            p1_dir = (p1_current[0] - p0[0], p1_current[1] - p0[1])
                            
                            if abs(p1_dir[0]) > 0.1 or abs(p1_dir[1]) > 0.1:
                                # Find scale that preserves midpoint
                                best_scale = SynchronizationHelper.find_handle_scale_for_midpoint(
                                    p0, p1_dir, p2_new, p3_new, target_mid
                                )
                                
                                new_p1 = (p0[0] + p1_dir[0] * best_scale, 
                                         p0[1] + p1_dir[1] * best_scale)
                                
                                # Find flat index for the opposite handle
                                opp_handle_idx = list(master_path.nodes).index(opp_handle)
                                opp_flat_idx = node_to_point.get((path_idx, opp_handle_idx))
                                
                                if opp_flat_idx is not None and opp_flat_idx < len(preview_points[master_idx]):
                                    preview_points[master_idx][opp_flat_idx] = (round(new_p1[0]), round(new_p1[1]))
                        
                        # Compensate "next" curve: p1 (our next handle) moved, adjust p2 (opposite handle)
                        # Curve: p0 (our node) -> p1 (our handle) -> p2 (opp_handle) -> p3 (next_oncurve)
                        if 'next' in curve_compensation and next_node.type == GSOFFCURVE:
                            info = curve_compensation['next']
                            opp_handle = info['opp_handle']
                            target_mid = info['midpoint']
                            anchor = info['anchor']  # p3, the next on-curve
                            
                            # New positions after ratio sync
                            p0_new = (new_pt_x, new_pt_y)  # Our node's new position
                            p1_new = (new_next_x, new_next_y)  # Our handle's new position
                            p3 = anchor
                            
                            # Direction from anchor (p3) to current opp_handle (p2)
                            p2_current = (opp_handle.position.x, opp_handle.position.y)
                            p2_dir = (p2_current[0] - p3[0], p2_current[1] - p3[1])
                            
                            if abs(p2_dir[0]) > 0.1 or abs(p2_dir[1]) > 0.1:
                                # Find scale that preserves midpoint
                                best_scale = SynchronizationHelper.find_handle_scale_for_midpoint_p2(
                                    p0_new, p1_new, p2_dir, p3, target_mid
                                )
                                
                                new_p2 = (p3[0] + p2_dir[0] * best_scale,
                                         p3[1] + p2_dir[1] * best_scale)
                                
                                # Find flat index for the opposite handle
                                opp_handle_idx = list(master_path.nodes).index(opp_handle)
                                opp_flat_idx = node_to_point.get((path_idx, opp_handle_idx))
                                
                                if opp_flat_idx is not None and opp_flat_idx < len(preview_points[master_idx]):
                                    preview_points[master_idx][opp_flat_idx] = (round(new_p2[0]), round(new_p2[1]))
                    
                except Exception:
                    continue
        
        # Convert lists back to GlyphCoordinates (GlyphCoordinates imported at top of file)
        result = {}
        for master_idx, points_list in preview_points.items():
            result[master_idx] = GlyphCoordinates(points_list)
        
        return result


# =============================================================================
# Dekink Panel - Non-modal floating panel using NSPanel directly
# =============================================================================

# Delegate class for handling NSPanel window events
# Using a separate NSObject subclass prevents Python reference cycles and crashes
class SyncRatiosPanelDelegate(NSObject):
    """Delegate for the NSPanel to handle window events safely."""
    
    def init(self):
        self = objc.super(SyncRatiosPanelDelegate, self).init()
        if self is None:
            return None
        self._panel_ref = None  # Will hold weak reference to SyncRatiosPanel
        return self
    
    def setPanel_(self, panel):
        """Set the panel reference (use weak ref to avoid cycles)."""
        import weakref
        self._panel_ref = weakref.ref(panel) if panel else None
    
    def windowWillClose_(self, notification):
        """Called when the window is about to close."""
        # Clear preview state synchronously before window fully closes
        SyncRatiosPanel._clear_preview_state()
        
        # Mark panel as closed
        panel = self._panel_ref() if self._panel_ref else None
        if panel:
            panel._is_closed = True
        
        # Clear class reference
        if SyncRatiosPanel._instance is panel:
            SyncRatiosPanel._instance = None
        
        # Trigger redraw to clear any preview
        try:
            Glyphs.redraw()
        except Exception:
            pass


# Action handler NSObject subclass - required because plain Python objects
# cannot be used as Objective-C targets for control actions
class SyncRatiosPanelActionHandler(NSObject):
    """Handles control actions for SyncRatiosPanel."""
    
    def init(self):
        self = objc.super(SyncRatiosPanelActionHandler, self).init()
        if self is None:
            return None
        self._panel_ref = None
        return self
    
    def setPanel_(self, panel):
        """Set the panel reference."""
        import weakref
        self._panel_ref = weakref.ref(panel) if panel else None
    
    def _getPanel(self):
        """Get the panel if still valid."""
        if self._panel_ref is None:
            return None
        panel = self._panel_ref()
        if panel is None or panel._is_closed:
            return None
        return panel
    
    @objc.typedSelector(b'v@:@')
    def checkboxAction_(self, sender):
        """Handle checkbox state changes."""
        panel = self._getPanel()
        if panel:
            panel._on_checkbox_changed(sender)
    
    @objc.typedSelector(b'v@:@')
    def radioAction_(self, sender):
        """Handle radio button state changes."""
        panel = self._getPanel()
        if panel:
            panel._on_radio_changed(sender)
    
    @objc.typedSelector(b'v@:@')
    def sliderAction_(self, sender):
        """Handle slider value changes."""
        panel = self._getPanel()
        if panel:
            panel._on_slider_changed(sender)
    
    @objc.typedSelector(b'v@:@')
    def syncButtonAction_(self, sender):
        """Handle Synchronize button click."""
        panel = self._getPanel()
        if panel:
            panel._on_sync_clicked(sender)


class SyncRatiosPanel:
    """
    Non-modal floating panel for synchronizing point ratios across masters.
    Uses NSPanel directly for better stability (avoids Vanilla callback issues).
    
    Allows the user to:
    - Choose between "Match Selection" and "Average All" modes
    - Adjust how much off-curves vs on-curves move
    - Preview changes in real-time before applying
    - Apply synchronization while still interacting with Glyphs
    """
    
    # Class-level reference to the single panel instance
    _instance = None
    
    # Class-level preview state (accessible by InterpolateState)
    _preview_active = False
    _preview_points = None  # Dict of master_idx -> GlyphCoordinates
    _preview_glyph_name = None  # Name of glyph being previewed
    _preview_content_hash = None  # Hash of glyph content when preview was created
    _preview_selection_hash = None  # Hash of selected nodes when preview was created
    
    @classmethod
    def show_panel(cls, font=None):
        """Show the Dekink panel, creating it if needed."""
        # If there's an existing instance, close it properly first
        if cls._instance is not None:
            try:
                old_instance = cls._instance
                cls._instance = None
                old_instance._close()
            except Exception:
                pass
        
        # Clear any stale preview state before creating new instance
        cls._clear_preview_state()
        
        # Create new instance
        cls._instance = cls()
        cls._instance._update_selection_count()
        cls._instance._update_preview()
        cls._instance._panel.makeKeyAndOrderFront_(None)
        cls._instance._position_window()
    
    @classmethod
    def _clear_preview_state(cls):
        """Clear all preview state. Called when preview becomes invalid."""
        cls._preview_active = False
        cls._preview_points = None
        cls._preview_glyph_name = None
        cls._preview_content_hash = None
        cls._preview_selection_hash = None
    
    @classmethod
    def invalidate_preview_if_stale(cls, glyph=None, font=None):
        """Check if the current preview is stale and invalidate if so.
        
        Should be called when glyph content may have changed (undo, edit, etc.)
        Returns True if preview was invalidated.
        """
        if not cls._preview_active:
            return False
        
        # If no glyph provided, try to get current glyph
        if glyph is None:
            try:
                if font is None:
                    font = Glyphs.font
                if font and font.selectedLayers:
                    glyph = font.selectedLayers[0].parent
            except Exception:
                pass
        
        if glyph is None:
            cls._clear_preview_state()
            return True
        
        # Check if glyph name matches
        if glyph.name != cls._preview_glyph_name:
            cls._clear_preview_state()
            return True
        
        # Check if content hash matches (detects undo/redo/edits)
        if cls._preview_content_hash is not None:
            try:
                if font is None:
                    font = Glyphs.font
                cache = InterpolationCache()
                current_hash = cache.get_glyph_content_hash(glyph, font)
                if current_hash != cls._preview_content_hash:
                    cls._clear_preview_state()
                    return True
            except Exception:
                pass
        
        return False
    
    @classmethod
    def close_panel(cls):
        """Close the panel if it exists."""
        # Clear preview state first
        cls._clear_preview_state()
        
        if cls._instance is not None:
            instance = cls._instance
            cls._instance = None
            try:
                instance._close()
            except Exception:
                pass
        
        try:
            Glyphs.redraw()
        except Exception:
            pass
    
    @classmethod
    def update_selection_if_open(cls):
        """Update the selection count if the panel is currently open."""
        if cls._instance is not None and not cls._instance._is_closed:
            try:
                cls._instance._update_selection_count()
                cls._instance._update_preview()
            except Exception:
                pass
    
    @classmethod
    def is_preview_active(cls):
        """Check if preview mode is currently active."""
        return cls._preview_active and cls._preview_points is not None
    
    @classmethod
    def get_preview_points(cls):
        """Get the current preview points if available."""
        return cls._preview_points
    
    @classmethod
    def get_preview_glyph_name(cls):
        """Get the name of the glyph being previewed."""
        return cls._preview_glyph_name
    
    def __init__(self):
        """Initialize the Dekink panel using NSPanel directly."""
        self._is_closed = False
        self._controls = {}  # Store references to controls
        
        # Create the panel with utility window style
        style = (NSWindowStyleMaskTitled | 
                 NSWindowStyleMaskClosable | 
                 NSWindowStyleMaskUtilityWindow)
        
        rect = NSMakeRect(100, 100, 220, 295)
        self._panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            style,
            NSBackingStoreBuffered,
            False
        )
        self._panel.setTitle_("Dekink")
        self._panel.setFloatingPanel_(True)
        self._panel.setBecomesKeyOnlyIfNeeded_(True)
        self._panel.setHidesOnDeactivate_(False)
        
        # Create and set the delegate for window events
        self._delegate = SyncRatiosPanelDelegate.alloc().init()
        self._delegate.setPanel_(self)
        self._panel.setDelegate_(self._delegate)
        
        # Create the action handler (NSObject subclass required for control targets)
        self._action_handler = SyncRatiosPanelActionHandler.alloc().init()
        self._action_handler.setPanel_(self)
        
        # Build the UI
        self._build_ui()
    
    def _build_ui(self):
        """Build the panel UI using AppKit controls directly."""
        content = self._panel.contentView()
        width = 220
        y = 265  # Start from top (we calculate from top going down)
        
        # Preview checkbox
        y -= 28
        self._controls['previewCheck'] = self._create_checkbox(
            content, "Preview dekink", (15, y, width - 30, 20), True
        )
        
        # Main mode radio buttons (Manual / Auto) - in their own container for grouping
        y -= 25
        mainModeHeight = 38
        mainModeContainer = NSView.alloc().initWithFrame_(NSMakeRect(15, y - mainModeHeight + 18, width - 30, mainModeHeight))
        content.addSubview_(mainModeContainer)
        self._controls['mainModeContainer'] = mainModeContainer
        
        self._controls['manualRadio'] = self._create_radio_button(
            mainModeContainer, "Manual Control", (0, 20, width - 30, 18), True
        )
        self._controls['autoRadio'] = self._create_radio_button(
            mainModeContainer, "Auto (Preserve Shape)", (0, 0, width - 30, 18), False
        )
        y -= mainModeHeight
        
        # --- Manual mode controls ---
        y -= 7
        self._controls['changeLabel'] = self._create_label(
            content, "Change:", (15, y, width - 30, 17)
        )
        
        # Balance slider
        y -= 25
        self._controls['balanceSlider'] = self._create_slider(
            content, (15, y, width - 30, 23), 0, 1, 0.15
        )
        
        # Slider labels
        y -= 18
        self._controls['onCurvesLabel'] = self._create_label(
            content, "On-Curves", (15, y, 80, 14), size="small"
        )
        self._controls['offCurvesLabel'] = self._create_label(
            content, "Off-Curves", (width - 95, y, 80, 14), size="small", alignment="right"
        )
        
        # Target mode radio buttons - in their own container for grouping
        y -= 25
        targetModeHeight = 38
        targetModeContainer = NSView.alloc().initWithFrame_(NSMakeRect(15, y - targetModeHeight + 18, width - 30, targetModeHeight))
        content.addSubview_(targetModeContainer)
        self._controls['targetModeContainer'] = targetModeContainer
        
        self._controls['matchRadio'] = self._create_radio_button(
            targetModeContainer, "Match Selection", (0, 20, width - 30, 18), True
        )
        self._controls['averageRadio'] = self._create_radio_button(
            targetModeContainer, "Average All", (0, 0, width - 30, 18), False
        )
        y -= targetModeHeight
        
        # Compensate checkbox
        y -= 7
        self._controls['compensateCheck'] = self._create_checkbox(
            content, "Compensate Curves", (15, y, width - 30, 20), True
        )
        
        # Synchronize button
        y -= 32
        self._controls['syncButton'] = self._create_button(
            content, "Apply Dekink", (15, y, width - 30, 24)
        )
        
        # Selection count label
        y -= 25
        self._controls['countLabel'] = self._create_label(
            content, "0 points selected", (15, y, width - 30, 17), size="small"
        )
        
        # Initial visibility update
        self._update_manual_controls_visibility()
    
    def _create_checkbox(self, parent, title, frame, checked):
        """Create a checkbox button."""
        button = NSButton.alloc().initWithFrame_(NSMakeRect(*frame))
        button.setButtonType_(NSButtonTypeSwitch)
        button.setTitle_(title)
        button.setState_(1 if checked else 0)
        button.setTarget_(self._action_handler)
        button.setAction_(self._action_handler.checkboxAction_)
        button.cell().setControlSize_(NSControlSizeSmall)
        button.setFont_(NSFont.systemFontOfSize_(11))
        parent.addSubview_(button)
        return button
    
    def _create_radio_button(self, parent, title, frame, selected):
        """Create a radio button."""
        button = NSButton.alloc().initWithFrame_(NSMakeRect(*frame))
        button.setButtonType_(4)  # NSRadioButton = 4
        button.setTitle_(title)
        button.setState_(1 if selected else 0)
        button.setTarget_(self._action_handler)
        button.setAction_(self._action_handler.radioAction_)
        button.cell().setControlSize_(NSControlSizeSmall)
        button.setFont_(NSFont.systemFontOfSize_(11))
        parent.addSubview_(button)
        return button
    
    def _create_button(self, parent, title, frame):
        """Create a push button."""
        button = NSButton.alloc().initWithFrame_(NSMakeRect(*frame))
        button.setButtonType_(NSButtonTypeMomentaryPushIn)
        button.setTitle_(title)
        button.setBezelStyle_(NSBezelStyleRounded)
        button.setTarget_(self._action_handler)
        button.setAction_(self._action_handler.syncButtonAction_)
        parent.addSubview_(button)
        return button
    
    def _create_slider(self, parent, frame, minVal, maxVal, value):
        """Create a slider."""
        slider = NSSlider.alloc().initWithFrame_(NSMakeRect(*frame))
        slider.setMinValue_(minVal)
        slider.setMaxValue_(maxVal)
        slider.setDoubleValue_(value)
        slider.setContinuous_(True)
        slider.setTarget_(self._action_handler)
        slider.setAction_(self._action_handler.sliderAction_)
        slider.cell().setControlSize_(NSControlSizeSmall)
        parent.addSubview_(slider)
        return slider
    
    def _create_label(self, parent, text, frame, size="regular", alignment="left"):
        """Create a text label."""
        label = NSTextField.alloc().initWithFrame_(NSMakeRect(*frame))
        label.setStringValue_(text)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        
        fontSize = 11 if size == "small" else 13
        label.setFont_(NSFont.systemFontOfSize_(fontSize))
        
        if alignment == "right":
            label.setAlignment_(NSTextAlignmentRight)
        elif alignment == "center":
            label.setAlignment_(NSTextAlignmentCenter)
        else:
            label.setAlignment_(NSTextAlignmentLeft)
        
        parent.addSubview_(label)
        return label
    
    def _update_manual_controls_visibility(self):
        """Show/hide manual controls based on mode selection."""
        if self._is_closed:
            return
        
        is_manual = self._controls.get('manualRadio') and self._controls['manualRadio'].state() == 1
        
        # Hide/show the target mode container (contains matchRadio and averageRadio)
        if 'targetModeContainer' in self._controls:
            self._controls['targetModeContainer'].setHidden_(not is_manual)
        
        controls_to_toggle = [
            'changeLabel', 'balanceSlider', 'onCurvesLabel', 'offCurvesLabel',
            'compensateCheck'
        ]
        
        for name in controls_to_toggle:
            if name in self._controls:
                self._controls[name].setHidden_(not is_manual)
    
    def _on_checkbox_changed(self, sender):
        """Handle checkbox state changes (called by action handler)."""
        if self._is_closed:
            return
        self._update_preview()
    
    def _on_radio_changed(self, sender):
        """Handle radio button state changes (called by action handler)."""
        if self._is_closed:
            return
        
        # Handle mutual exclusivity for main mode (Manual/Auto)
        if sender == self._controls.get('manualRadio'):
            self._controls['autoRadio'].setState_(0)
            self._controls['manualRadio'].setState_(1)
            self._update_manual_controls_visibility()
        elif sender == self._controls.get('autoRadio'):
            self._controls['manualRadio'].setState_(0)
            self._controls['autoRadio'].setState_(1)
            self._update_manual_controls_visibility()
        
        # Handle mutual exclusivity for target mode (Match/Average)
        if sender == self._controls.get('matchRadio'):
            self._controls['averageRadio'].setState_(0)
            self._controls['matchRadio'].setState_(1)
        elif sender == self._controls.get('averageRadio'):
            self._controls['matchRadio'].setState_(0)
            self._controls['averageRadio'].setState_(1)
        
        self._update_preview()
    
    def _on_slider_changed(self, sender):
        """Handle slider value changes (called by action handler)."""
        if self._is_closed:
            return
        self._update_preview()
    
    def _on_sync_clicked(self, sender):
        """Handle Synchronize button click (called by action handler)."""
        if self._is_closed:
            return
        
        try:
            font = Glyphs.font
            if not font or not font.selectedLayers:
                return
            
            layer = font.selectedLayers[0]
            glyph = layer.parent
            if not glyph:
                return
            
            selected_nodes = SynchronizationHelper.get_selected_smooth_oncurve_nodes(layer)
            if not selected_nodes:
                return
            
            # Filter to only nodes with kinks (or curve-segment counterparts)
            filtered_nodes = SynchronizationHelper.filter_nodes_with_kinks(selected_nodes, layer, font)
            if not filtered_nodes:
                return  # No kinked nodes to sync
            
            is_auto_mode = self._controls['autoRadio'].state() == 1
            
            font.disableUpdateInterface()
            
            try:
                if is_auto_mode:
                    synced_count = SynchronizationHelper.sync_ratios_auto(
                        filtered_nodes, font, max_deviation=25.0
                    )
                else:
                    mode = "follow" if self._controls['matchRadio'].state() == 1 else "average"
                    off_curve_change = self._controls['balanceSlider'].doubleValue()
                    compensate_curves = self._controls['compensateCheck'].state() == 1
                    
                    synced_count = SynchronizationHelper.sync_ratios(
                        filtered_nodes, font, mode=mode, off_curve_change=off_curve_change,
                        compensate_curves=compensate_curves
                    )
                
                if synced_count > 0:
                    for master in font.masters:
                        master_layer = glyph.layers[master.id]
                        if master_layer:
                            master_layer.syncMetrics()
            finally:
                font.enableUpdateInterface()
            
            Glyphs.redraw()
            
            # Update kink detection
            state = InterpolateState.get_state_for_font(font)
            if state:
                state.clear_caches(clear_kinks=True)
                state._detect_potential_kinks()
                state.update_glyph()
            
        except Exception as e:
            if DEBUG_INTERPOLATE:
                print(f"SyncRatiosPanel._on_sync_clicked error: {e}")
                import traceback
                traceback.print_exc()
    
    def _update_selection_count(self):
        """Update the selection count label."""
        if self._is_closed:
            return
        
        try:
            font = Glyphs.font
            if not font or not font.selectedLayers:
                self._controls['countLabel'].setStringValue_("No selection")
                return
            
            layer = font.selectedLayers[0]
            selected_nodes = SynchronizationHelper.get_selected_smooth_oncurve_nodes(layer)
            total_count = len(selected_nodes)
            
            if total_count == 0:
                self._controls['countLabel'].setStringValue_("No smooth points selected")
                return
            
            # Count how many have kinks
            filtered_nodes = SynchronizationHelper.filter_nodes_with_kinks(selected_nodes, layer, font)
            kinked_count = len(filtered_nodes)
            
            if kinked_count == 0:
                self._controls['countLabel'].setStringValue_(f"{total_count} selected, 0 with kinks")
            elif kinked_count == total_count:
                if kinked_count == 1:
                    self._controls['countLabel'].setStringValue_("1 point with kink")
                else:
                    self._controls['countLabel'].setStringValue_(f"{kinked_count} points with kinks")
            else:
                self._controls['countLabel'].setStringValue_(f"{kinked_count} of {total_count} with kinks")
        except Exception:
            try:
                self._controls['countLabel'].setStringValue_("Error getting selection")
            except Exception:
                pass
    
    def _update_preview(self):
        """Update the preview based on current settings."""
        if self._is_closed:
            return
        
        try:
            preview_enabled = self._controls['previewCheck'].state() == 1
            
            if not preview_enabled:
                SyncRatiosPanel._clear_preview_state()
                Glyphs.redraw()
                return
            
            font = Glyphs.font
            if not font or not font.selectedLayers:
                SyncRatiosPanel._clear_preview_state()
                return
            
            layer = font.selectedLayers[0]
            glyph = layer.parent
            if not glyph:
                SyncRatiosPanel._clear_preview_state()
                return
            
            selected_nodes = SynchronizationHelper.get_selected_smooth_oncurve_nodes(layer)
            if not selected_nodes:
                SyncRatiosPanel._clear_preview_state()
                Glyphs.redraw()
                return
            
            # Filter to only nodes with kinks (or curve-segment counterparts)
            filtered_nodes = SynchronizationHelper.filter_nodes_with_kinks(selected_nodes, layer, font)
            if not filtered_nodes:
                SyncRatiosPanel._clear_preview_state()
                Glyphs.redraw()
                return
            
            is_auto_mode = self._controls['autoRadio'].state() == 1
            
            if is_auto_mode:
                preview_points = SynchronizationHelper.calculate_preview_points_auto(
                    filtered_nodes, font, max_deviation=25.0
                )
            else:
                mode = "follow" if self._controls['matchRadio'].state() == 1 else "average"
                off_curve_change = self._controls['balanceSlider'].doubleValue()
                compensate_curves = self._controls['compensateCheck'].state() == 1
                
                preview_points = SynchronizationHelper.calculate_preview_points(
                    filtered_nodes, font, mode=mode, off_curve_change=off_curve_change,
                    compensate_curves=compensate_curves
                )
            
            if preview_points:
                SyncRatiosPanel._preview_active = True
                SyncRatiosPanel._preview_points = preview_points
                SyncRatiosPanel._preview_glyph_name = glyph.name
                
                try:
                    cache = InterpolationCache()
                    SyncRatiosPanel._preview_content_hash = cache.get_glyph_content_hash(glyph, font)
                except Exception:
                    SyncRatiosPanel._preview_content_hash = None
                
                try:
                    selection_parts = []
                    for node in filtered_nodes:
                        selection_parts.append(id(node))
                    SyncRatiosPanel._preview_selection_hash = tuple(selection_parts)
                except Exception:
                    SyncRatiosPanel._preview_selection_hash = None
            else:
                SyncRatiosPanel._clear_preview_state()
            
            Glyphs.redraw()
            
        except Exception as e:
            if DEBUG_INTERPOLATE:
                print(f"SyncRatiosPanel._update_preview error: {e}")
                import traceback
                traceback.print_exc()
            SyncRatiosPanel._clear_preview_state()
    
    def _position_window(self):
        """Position the window in the lower-right area of the screen."""
        try:
            screen = NSScreen.mainScreen()
            if not screen:
                return
            
            screen_frame = screen.visibleFrame()
            screen_width = screen_frame.size.width
            screen_height = screen_frame.size.height
            screen_x = screen_frame.origin.x
            screen_y = screen_frame.origin.y
            
            panel_frame = self._panel.frame()
            window_width = panel_frame.size.width
            window_height = panel_frame.size.height
            
            x = screen_x + screen_width * 0.75 - window_width / 2
            y = screen_y + screen_height * 0.25
            
            self._panel.setFrameOrigin_((x, y))
        except Exception:
            pass
    
    def _close(self):
        """Close the panel safely."""
        if self._is_closed:
            return
        
        self._is_closed = True
        SyncRatiosPanel._clear_preview_state()
        
        try:
            # Remove delegate first to prevent callbacks during close
            if self._panel:
                self._panel.setDelegate_(None)
            if self._delegate:
                self._delegate.setPanel_(None)
            if self._action_handler:
                self._action_handler.setPanel_(None)
            
            # Close the panel
            if self._panel:
                self._panel.close()
        except Exception:
            pass
        
        self._panel = None
        self._delegate = None
        self._action_handler = None
        self._controls = {}


# =============================================================================
# Configuration Constants
# =============================================================================

# Default colors (RGBA tuples) - used as fallback when no user preference is set
DEFAULT_FILL_COLOR = (0.38, 0.0, 0.88, 0.15)
DEFAULT_OUTLINE_COLOR = (0.0, 0.0, 0.0, 0.8)  # Used for outline, nodes, and handles
DEFAULT_KINK_INDICATOR_COLOR = (1.0, 0.0, 0.6, 1.0)   # Used for kink indicators
DEFAULT_DEKINK_PREVIEW_COLOR = (0.0, 0.7, 0.2, 1.0)   # Used for dekink window preview (green)
DEFAULT_TOOL_SHORTCUT = 'a'
DEFAULT_USE_DOTTED_LINES = True  # True = dotted, False = solid


def _load_color_from_defaults(key: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    """Load a color tuple from NSUserDefaults, falling back to default if not set."""
    defaults = NSUserDefaults.standardUserDefaults()
    stored = defaults.objectForKey_(key)
    if stored is not None:
        try:
            # Stored as array of floats
            if len(stored) >= 3:
                if len(stored) == 3:
                    return (float(stored[0]), float(stored[1]), float(stored[2]), 1.0)
                return (float(stored[0]), float(stored[1]), float(stored[2]), float(stored[3]))
        except Exception:
            pass
    return default


def _save_color_to_defaults(key: str, color: Tuple[float, ...]) -> None:
    """Save a color tuple to NSUserDefaults."""
    defaults = NSUserDefaults.standardUserDefaults()
    defaults.setObject_forKey_(list(color), key)
    try:
        defaults.synchronize()
    except Exception:
        pass


class InterpolateConfig:
    """Centralized configuration constants for the Interpolate plugin.
    
    Colors and some settings can be customized via the Interpol Settings window.
    The getters read from NSUserDefaults with fallback to hardcoded defaults.
    """
    __slots__ = ()  # Prevent accidental instance attribute creation
    
    # Kink detection thresholds
    KINK_VISIBILITY_THRESHOLD = 0.01  # Min severity to show a kink indicator
    
    # Mouse sensitivity for tool drag (pixels per full axis traverse)
    TOOL_MOUSE_SENSITIVITY = 400.0
    
    # Color getters that read from preferences
    @staticmethod
    def get_fill_color() -> Tuple[float, ...]:
        """Get the preview fill color from preferences."""
        return _load_color_from_defaults(KEY + ".color.fill", DEFAULT_FILL_COLOR)
    
    @staticmethod
    def get_outline_color() -> Tuple[float, ...]:
        """Get the preview outline/node/handle color from preferences."""
        return _load_color_from_defaults(KEY + ".color.outline", DEFAULT_OUTLINE_COLOR)
    
    @staticmethod
    def get_kink_indicator_color() -> Tuple[float, ...]:
        """Get the kink indicator color from preferences."""
        return _load_color_from_defaults(KEY + ".color.kink_indicator", DEFAULT_KINK_INDICATOR_COLOR)
    
    @staticmethod
    def get_dekink_preview_color() -> Tuple[float, ...]:
        """Get the dekink window preview color from preferences."""
        return _load_color_from_defaults(KEY + ".color.dekink_preview", DEFAULT_DEKINK_PREVIEW_COLOR)
    
    @staticmethod
    def get_dekink_preview_colors() -> Dict[str, Tuple[float, ...]]:
        """Get derived colors for the dekink window preview based on the base color.
        
        Returns a dict with keys: fill, stroke, node, handle, handle_line
        """
        base = InterpolateConfig.get_dekink_preview_color()
        r, g, b = base[0], base[1], base[2]
        return {
            'fill': (r, g + 0.1, b + 0.1, 0.25),       # Lighter fill with transparency
            'stroke': (r, g, b, 0.9),                   # Full stroke
            'node': (r, g, b, 1.0),                     # Full opacity nodes
            'handle': (r, g, b, 0.8),                   # Slightly transparent handles
            'handle_line': (r, g, b, 0.5),              # More transparent handle lines
        }
    
    @staticmethod
    def get_tool_shortcut() -> str:
        """Get the tool keyboard shortcut from preferences."""
        defaults = NSUserDefaults.standardUserDefaults()
        stored = defaults.stringForKey_(KEY + ".tool_shortcut")
        return stored if stored else DEFAULT_TOOL_SHORTCUT
    
    @staticmethod
    def get_use_dotted_lines() -> bool:
        """Get whether to use dotted lines (True) or solid lines (False)."""
        defaults = NSUserDefaults.standardUserDefaults()
        if defaults.objectForKey_(KEY + ".use_dotted_lines") is not None:
            return defaults.boolForKey_(KEY + ".use_dotted_lines")
        return DEFAULT_USE_DOTTED_LINES
    
    # Static fallback colors for things that don't need to be configurable
    FILL_COLOR = DEFAULT_FILL_COLOR  # Kept for backwards compatibility
    NODE_COLOR = DEFAULT_OUTLINE_COLOR
    HANDLE_COLOR = DEFAULT_OUTLINE_COLOR
    KINK_COLOR = (1.0, 0.0, 0.6)
    KINK_INNER_COLOR = (1.0, 0.6, 0.8)
    ANCHOR_COLOR = (1.0, 0.1, 0.1, 0.9)
    BUBBLE_BORDER_COLOR = (0.7, 0.0, 0.9, 1.0)
    # Fallback SYNC_PREVIEW colors kept for backwards compatibility
    SYNC_PREVIEW_FILL_COLOR = (0.0, 0.8, 0.3, 0.25)
    SYNC_PREVIEW_STROKE_COLOR = (0.0, 0.7, 0.2, 0.9)
    SYNC_PREVIEW_NODE_COLOR = (0.0, 0.7, 0.2, 1.0)      # Green on-curve points
    SYNC_PREVIEW_HANDLE_COLOR = (0.0, 0.7, 0.2, 0.8)    # Green off-curve handles
    SYNC_PREVIEW_HANDLE_LINE_COLOR = (0.0, 0.7, 0.2, 0.5)  # Green handle lines
    
    # UI sizing
    BUBBLE_CORNER_RADIUS = 25
    BUBBLE_VERTICAL_MARGIN = 30
    BUBBLE_HORIZONTAL_MARGIN = 100
    
    # Preview update throttling (~60fps)
    PREVIEW_UPDATE_INTERVAL = 0.016


# =============================================================================
# Cache Management
# =============================================================================

class InterpolationCache:
    """
    Centralized cache management for interpolation data.
    
    Provides automatic invalidation based on font, glyph, and scalars changes.
    All caches are accessed through a single interface for consistency.
    """
    
    def __init__(self):
        # Glyph point structures (keyed by glyph_name -> {content_hash, points, widths, ...})
        self._glyph_points = {}
        
        # Interpolated paths (keyed by (glyph_name, scalars_hash, content_hash))
        self._paths = {}
        
        # Font-level caches
        self._font_bounds = None  # (minY, maxY)
        self._font_bounds_id = None  # Font signature for validation
        self._font_metrics = None  # (descender, ascender, upm)
        self._font_metrics_id = None  # Font signature for metrics validation
        
        # Kink tracking (keyed by node_id -> max_severity)
        self._kink_severities = {}
        
        # Track last scalars for change detection
        self._last_scalars_hash = None
    
    def get_scalars_hash(self, master_scalars: List[float]) -> Optional[Tuple[float, ...]]:
        """Get a hashable representation of master scalars."""
        if not master_scalars:
            return None
        return tuple(round(s, 6) for s in master_scalars)
    
    def get_glyph_content_hash(self, glyph: Any, font: Any) -> Optional[Tuple]:
        """
        Get a hash of glyph content across all masters to detect edits.
        Returns a hashable tuple representing the glyph's current state.
        """
        if not glyph or not font:
            return None
        try:
            hash_parts = []
            for master in font.masters:
                layer = glyph.layers[master.id]
                if layer:
                    # Include width and path count as quick hash
                    hash_parts.append(layer.width)
                    hash_parts.append(len(layer.paths))
                    # Include point count per path
                    for path in layer.paths:
                        hash_parts.append(len(path.nodes))
                        # Include a few key coordinates for finer change detection
                        for node in path.nodes[:3]:  # First 3 nodes
                            hash_parts.append(round(node.x, 1))
                            hash_parts.append(round(node.y, 1))
                    # Include component info
                    hash_parts.append(len(layer.components))
                    for comp in layer.components:
                        hash_parts.append(comp.componentName)
                        hash_parts.append(round(comp.x, 1))
                        hash_parts.append(round(comp.y, 1))
            return tuple(hash_parts)
        except Exception:
            return None
    
    def get_cached_glyph_points(self, glyph_name: str) -> Optional[Dict]:
        """Get cached glyph point data if available."""
        return self._glyph_points.get(glyph_name)
    
    def set_glyph_points(self, glyph_name: str, data: Dict) -> None:
        """Store glyph point data in cache."""
        self._glyph_points[glyph_name] = data
    
    def get_cached_path(self, glyph_name: str, scalars_hash: Optional[Tuple], content_hash: Optional[Tuple]) -> Optional[Tuple]:
        """Get cached interpolated path if available."""
        cache_key = (glyph_name, scalars_hash, content_hash)
        return self._paths.get(cache_key)
    
    def set_path(self, glyph_name: str, scalars_hash: Optional[Tuple], content_hash: Optional[Tuple], path_data: Tuple) -> None:
        """Store interpolated path in cache, cleaning old entries for this glyph."""
        cache_key = (glyph_name, scalars_hash, content_hash)
        # Clean up old cache entries for this glyph
        keys_to_remove = [k for k in self._paths if k[0] == glyph_name and k != cache_key]
        for k in keys_to_remove:
            del self._paths[k]
        self._paths[cache_key] = path_data
    
    def get_font_bounds(self, font_signature: str) -> Optional[Tuple[float, float]]:
        """Get cached font bounds if valid for given signature."""
        if self._font_bounds_id == font_signature:
            return self._font_bounds
        return None
    
    def set_font_bounds(self, font_signature: str, bounds: Tuple[float, float]) -> None:
        """Store font bounds with signature for validation."""
        self._font_bounds = bounds
        self._font_bounds_id = font_signature
    
    def get_font_metrics(self, font_signature: str) -> Optional[Tuple[float, float, float]]:
        """Get cached font metrics (descender, ascender, upm) if valid for given signature."""
        if self._font_metrics_id == font_signature:
            return self._font_metrics
        return None
    
    def set_font_metrics(self, font_signature: str, metrics: Tuple[float, float, float]) -> None:
        """Store font metrics (descender, ascender, upm) with signature for validation."""
        self._font_metrics = metrics
        self._font_metrics_id = font_signature

    
    def get_kink_severity(self, node_id: str) -> float:
        """Get max severity recorded for a kink node."""
        return self._kink_severities.get(node_id, 0)
    
    def update_kink_severity(self, node_id: str, severity: float) -> None:
        """Update max severity for a kink node if higher than recorded."""
        current = self._kink_severities.get(node_id, 0)
        if severity > current:
            self._kink_severities[node_id] = severity
    
    def invalidate_glyph(self, glyph_name=None, clear_kinks=False):
        """Invalidate caches for a specific glyph or all glyphs.
        
        Args:
            glyph_name: Specific glyph to invalidate, or None for all
            clear_kinks: If True, also clear kink severity tracking.
                         Should be True when switching to a different glyph,
                         False when just updating interpolation position.
        """
        if glyph_name:
            self._glyph_points.pop(glyph_name, None)
            keys_to_remove = [k for k in self._paths if k[0] == glyph_name]
            for k in keys_to_remove:
                del self._paths[k]
        else:
            self._glyph_points.clear()
            self._paths.clear()
        # Only clear kink severities when explicitly requested (e.g., glyph changed)
        if clear_kinks:
            self._kink_severities.clear()
        self._last_scalars_hash = None

    def invalidate_all(self):
        """Clear all caches completely."""
        self._glyph_points.clear()
        self._paths.clear()
        self._font_bounds = None
        self._font_bounds_id = None
        self._font_metrics = None
        self._font_metrics_id = None
        self._kink_severities.clear()
        self._last_scalars_hash = None


# =============================================================================
# Glyph Interpolation Engine
# =============================================================================

class GlyphInterpolator:
    """
    Handles glyph point extraction and interpolation.
    
    This is the core engine for computing interpolated glyph shapes.
    It extracts points from master layers, interpolates them using
    the variation model, and builds NSBezierPath objects for rendering.
    """
    
    @staticmethod
    def extract_points_from_layer(layer: Any) -> Any:
        """
        Extract coordinate points from a layer's paths.
        
        Args:
            layer: A GSLayer (preferably decomposed to include components)
            
        Returns:
            GlyphCoordinates object containing all path points
        """
        points = []
        for path in layer.paths:
            for seg_ix, seg in enumerate(path.segments):
                if seg_ix == 0:
                    points.append((seg[0].x, seg[0].y))
                points.append((seg[1].x, seg[1].y))
                if len(seg) == 4:
                    points.append((seg[2].x, seg[2].y))
                    points.append((seg[3].x, seg[3].y))
        return GlyphCoordinates(points)
    
    @staticmethod
    def build_node_to_point_map(layer: Any) -> Dict[Tuple[int, int], int]:
        """
        Build a mapping from (path_index, node_index) to flat index in extracted points.
        
        This maps from the path.nodes order to the segment-based extraction order
        used by extract_points_from_layer and interpolate_points.
        
        Args:
            layer: A GSLayer
            
        Returns:
            Dict mapping (path_idx, node_idx) to flat point index
        """
        node_to_point = {}
        flat_idx = 0
        
        for path_idx, path in enumerate(layer.paths):
            # Build a map from node position to node index in path.nodes
            # Use position tuple as key since segment nodes may be different objects
            node_pos_to_idx = {}
            for node_idx, node in enumerate(path.nodes):
                pos_key = (round(node.position.x, 2), round(node.position.y, 2))
                node_pos_to_idx[pos_key] = node_idx
            
            for seg_ix, seg in enumerate(path.segments):
                if seg_ix == 0:
                    # First segment includes start point
                    pos_key = (round(seg[0].x, 2), round(seg[0].y, 2))
                    node_idx = node_pos_to_idx.get(pos_key)
                    if node_idx is not None:
                        node_to_point[(path_idx, node_idx)] = flat_idx
                    flat_idx += 1
                
                # All segments include their non-start points
                for pt_ix in range(1, len(seg)):
                    pos_key = (round(seg[pt_ix].x, 2), round(seg[pt_ix].y, 2))
                    node_idx = node_pos_to_idx.get(pos_key)
                    if node_idx is not None:
                        node_to_point[(path_idx, node_idx)] = flat_idx
                    flat_idx += 1
        
        return node_to_point
    
    @staticmethod
    def get_decomposed_layer(layer: Any) -> Any:
        """
        Get a decomposed version of a layer (flattens components).
        
        Args:
            layer: A GSLayer
            
        Returns:
            Decomposed GSLayer, or original if decomposition fails
        """
        try:
            return layer.copyDecomposedLayer()
        except Exception:
            return layer
    
    @staticmethod
    def extract_master_data(
        glyph: Any,
        font: Any
    ) -> Tuple[Optional[Dict[int, Any]], Optional[List[float]], Optional[Any]]:
        """
        Extract point data and widths from all master layers of a glyph.
        
        Args:
            glyph: A GSGlyph
            font: The GSFont containing the glyph
            
        Returns:
            tuple: (glyph_points_dict, widths_list, reference_layer) or (None, None, None)
        """
        if not glyph or not font or not font.masters:
            return None, None, None
        
        master_layers = []
        decomposed_layers = []
        
        for master in font.masters:
            layer = glyph.layers[master.id]
            if layer:
                master_layers.append(layer)
                decomposed_layers.append(GlyphInterpolator.get_decomposed_layer(layer))
        
        if not master_layers:
            return None, None, None
        
        glyph_points = {}
        widths = []
        
        for ix, decomposed in enumerate(decomposed_layers):
            glyph_points[ix] = GlyphInterpolator.extract_points_from_layer(decomposed)
            widths.append(master_layers[ix].width)
        
        return glyph_points, widths, decomposed_layers[0]
    
    @staticmethod
    def validate_points_compatibility(glyph_points: Dict[int, Any]) -> bool:
        """
        Check if all master point arrays have the same length (compatible for interpolation).
        
        Args:
            glyph_points: dict mapping master index to GlyphCoordinates
            
        Returns:
            bool: True if all masters have compatible point counts
        """
        if not glyph_points:
            return False
        all_points = list(glyph_points.values())
        if not all_points:
            return False
        first_len = len(all_points[0])
        return all(len(points) == first_len for points in all_points)
    
    @staticmethod
    def interpolate_points(
        glyph_points: Dict[int, Any],
        model: Any,
        master_scalars: List[float]
    ) -> Optional[Any]:
        """
        Interpolate glyph points using the variation model.
        
        Args:
            glyph_points: dict mapping master index to GlyphCoordinates
            model: VariationModel instance
            master_scalars: list of scalar values for each master
            
        Returns:
            Interpolated coordinates or None if incompatible
        """
        all_points = list(glyph_points.values())
        if len(master_scalars) != len(all_points):
            return None
        return model.interpolateFromValuesAndScalars(all_points, master_scalars)
    
    @staticmethod
    def interpolate_width(widths: List[float], master_scalars: List[float]) -> float:
        """
        Interpolate glyph width from master widths.
        
        Args:
            widths: list of widths for each master
            master_scalars: list of scalar values for each master
            
        Returns:
            Interpolated width value
        """
        return sum(w * s for w, s in zip(widths, master_scalars))
    
    @staticmethod
    def build_bezier_path(
        reference_layer: Any,
        interpolated_points: Any
    ) -> Tuple[Any, List[Tuple[bool, Any]], List[Tuple[Any, Any]]]:
        """
        Build an NSBezierPath from interpolated points using the reference layer structure.
        
        Args:
            reference_layer: A GSLayer providing the path/segment structure
            interpolated_points: Iterable of (x, y) coordinate tuples
            
        Returns:
            tuple: (NSBezierPath, interp_nodes, interp_handles)
                - interp_nodes: list of (is_oncurve, NSPoint) tuples
                - interp_handles: list of (handle_point, connected_oncurve) tuples
        """
        displaypath = NSBezierPath.alloc().init()
        interp_nodes = []
        interp_handles = []
        point_iter = iter(interpolated_points)
        
        for path in reference_layer.paths:
            for ix, seg in enumerate(path.segments):
                if ix == 0:
                    move = NSPoint(*next(point_iter))
                    displaypath.moveToPoint_(move)
                    interp_nodes.append((True, move))
                    prev_point = move
                
                if len(seg) == 2:
                    line = NSPoint(*next(point_iter))
                    displaypath.lineToPoint_(line)
                    interp_nodes.append((True, line))
                    prev_point = line
                elif len(seg) == 3:
                    cp1 = NSPoint(*next(point_iter))
                    dest = NSPoint(*next(point_iter))
                    displaypath.curveToPoint_controlPoint_(dest, cp1)
                    interp_handles.append((cp1, prev_point))
                    interp_nodes.append((True, dest))
                    prev_point = dest
                else:
                    cp1 = NSPoint(*next(point_iter))
                    cp2 = NSPoint(*next(point_iter))
                    dest = NSPoint(*next(point_iter))
                    displaypath.curveToPoint_controlPoint1_controlPoint2_(dest, cp1, cp2)
                    interp_handles.append((cp1, prev_point))
                    interp_handles.append((cp2, dest))
                    interp_nodes.append((True, dest))
                    prev_point = dest
        
        return displaypath, interp_nodes, interp_handles
    
    @staticmethod
    def build_bezier_path_simple(reference_layer: Any, interpolated_points: Any) -> Optional[Any]:
        """
        Build an NSBezierPath from interpolated points (simplified version without node tracking).
        
        Args:
            reference_layer: A GSLayer providing the path/segment structure
            interpolated_points: Iterable of (x, y) coordinate tuples
            
        Returns:
            NSBezierPath or None if building fails
        """
        try:
            displaypath = NSBezierPath.alloc().init()
            point_iter = iter(interpolated_points)
            
            for path in reference_layer.paths:
                for ix, seg in enumerate(path.segments):
                    if ix == 0:
                        displaypath.moveToPoint_(NSPoint(*next(point_iter)))
                    
                    if len(seg) == 2:
                        displaypath.lineToPoint_(NSPoint(*next(point_iter)))
                    elif len(seg) == 3:
                        cp1 = NSPoint(*next(point_iter))
                        dest = NSPoint(*next(point_iter))
                        displaypath.curveToPoint_controlPoint_(dest, cp1)
                    else:
                        cp1 = NSPoint(*next(point_iter))
                        cp2 = NSPoint(*next(point_iter))
                        dest = NSPoint(*next(point_iter))
                        displaypath.curveToPoint_controlPoint1_controlPoint2_(dest, cp1, cp2)
            
            return displaypath
        except Exception:
            return None
    
    @staticmethod
    def interpolate_anchors(
        glyph: Any,
        font: Any,
        master_scalars: List[float]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Interpolate anchor positions for a glyph.
        
        Args:
            glyph: A GSGlyph
            font: The GSFont
            master_scalars: list of scalar values for each master
            
        Returns:
            dict mapping anchor_name to (x, y) interpolated position
        """
        result = {}
        if not glyph or not font or not master_scalars:
            return result
        
        try:
            # Get all master layers
            master_layers = []
            for master in font.masters:
                layer = glyph.layers[master.id]
                if layer:
                    master_layers.append(layer)
            
            if len(master_layers) != len(master_scalars):
                return result
            
            # Find common anchor names across all masters
            anchor_names_sets = [set(a.name for a in layer.anchors) for layer in master_layers]
            if not anchor_names_sets:
                return result
            common_anchor_names = set.intersection(*anchor_names_sets)
            
            # Interpolate each anchor
            for anchor_name in common_anchor_names:
                interp_x = 0
                interp_y = 0
                for layer, scalar in zip(master_layers, master_scalars):
                    for a in layer.anchors:
                        if a.name == anchor_name:
                            interp_x += a.x * scalar
                            interp_y += a.y * scalar
                            break
                result[anchor_name] = (interp_x, interp_y)
        except Exception:
            pass
        
        return result
    
    @staticmethod
    def get_interpolated_kerning(
        left_glyph: Any,
        right_glyph: Any,
        font: Any,
        master_scalars: List[float]
    ) -> float:
        """
        Get interpolated kerning value between two glyphs.
        
        Args:
            left_glyph: The left glyph of the pair
            right_glyph: The right glyph of the pair
            font: The GSFont
            master_scalars: list of scalar values for each master
            
        Returns:
            Interpolated kerning value (float)
        """
        if not master_scalars or not font:
            return 0
        
        try:
            # Use kerning keys
            left_key = left_glyph.rightKerningKey
            right_key = right_glyph.leftKerningKey
            
            # Collect kerning values from all masters
            kern_values = []
            for master in font.masters:
                kern = font.kerningForPair(master.id, left_key, right_key)
                kern_values.append(kern if kern is not None else 0)
            
            if len(kern_values) != len(master_scalars):
                return 0
            
            return sum(k * s for k, s in zip(kern_values, master_scalars))
        except Exception:
            return 0


# =============================================================================
# Kink Detection
# =============================================================================

class KinkDetector:
    """
    Detects potential kink nodes in glyph outlines by comparing angles and
    handle proportions across masters.
    
    A node is marked as a potential kink if:
    - The angles across masters differ by more than 1.0 degree, AND
    - The handle proportions across masters differ by more than 0.5%
    
    (Both conditions must be true - this matches showAngleProportionKink.py)
    """
    
    @staticmethod
    def dot_product(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """Calculate dot product of two 2D vectors."""
        return v1[0] * v2[0] + v1[1] * v2[1]
    
    @staticmethod
    def normalize_vector(v: Tuple[float, float]) -> Tuple[float, float]:
        """Normalize a 2D vector to unit length."""
        length = math.sqrt(v[0] ** 2 + v[1] ** 2)
        if length == 0:
            return (0, 0)
        return (v[0] / length, v[1] / length)
    
    @staticmethod
    def severity_to_degrees(severity: float) -> float:
        """
        Convert kink severity (0-200) to angle in degrees (0-180).
        Severity = (1 - dot_product) * 100
        dot_product = cos(angle), so angle = acos(1 - severity/100)
        """
        dp = 1 - (severity / 100.0)
        dp = max(-1.0, min(1.0, dp))  # Clamp to valid acos range
        angle_rad = math.acos(dp)
        return math.degrees(angle_rad)
    
    @staticmethod
    def calculate_kink_severity(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """
        Calculate kink severity at point p2 given neighbors p1 and p3.
        
        Returns severity 0-200 where:
        - 0 = perfectly smooth (collinear)
        - 100 = 90 degree kink
        - 200 = 180 degree kink (complete reversal)
        """
        # Direction vectors
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        v1_norm = KinkDetector.normalize_vector(v1)
        v2_norm = KinkDetector.normalize_vector(v2)
        
        if v1_norm == (0, 0) or v2_norm == (0, 0):
            return 0.0
        
        # Dot product: 1.0 = collinear (smooth), -1.0 = opposite (max kink)
        dp = KinkDetector.dot_product(v1_norm, v2_norm)
        severity = (1 - dp) * 100
        
        return severity
    
    @staticmethod
    def hypotenuse(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate distance between two points."""
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
    
    @staticmethod
    def get_angle(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate angle between two points in degrees."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx))
        return round(angle, 1)
    
    @staticmethod
    def get_prev_next_nodes(path, node_index: int):
        """Get previous and next nodes for a given node index."""
        nodes = path.nodes
        num_nodes = len(nodes)
        prev_node = nodes[(node_index - 1) % num_nodes]
        next_node = nodes[(node_index + 1) % num_nodes]
        return prev_node, next_node
    
    @staticmethod
    def get_layer_ids(layer, font) -> List[str]:
        """Get the master and special layer IDs to check for kinks."""
        layer_ids = set()
        glyph = layer.parent
        
        # Get "Ignore Kinks Along Axes" custom parameter
        ignore_axes = []
        if "Ignore Kinks Along Axes" in font.customParameters:
            ignore_param = font.customParameters["Ignore Kinks Along Axes"]
            if ignore_param:
                ignore_axes = [x.strip() for x in ignore_param.split(",")]
                axes_tags = [axis.axisTag for axis in font.axes]
                ignore_axes = [x for x in ignore_axes if x in axes_tags]
        
        active_master = layer.master
        active_master_coords = list(active_master.axes) if active_master else []
        
        axes_tags = [axis.axisTag for axis in font.axes]
        
        def match_ignored_axes(lyr):
            """Check if layer matches active master on ignored axes."""
            if not ignore_axes:
                return True
            lyr_coords = []
            if lyr.layerId == lyr.associatedMasterId:
                # Master layer
                for m in font.masters:
                    if m.id == lyr.layerId:
                        lyr_coords = list(m.axes)
                        break
            elif lyr.isBraceLayer():
                coords_dict = lyr.attributes.get("coordinates", {})
                if coords_dict:
                    for axis in font.axes:
                        lyr_coords.append(coords_dict.get(axis.axisId, 0))
            
            if not lyr_coords:
                return False
            
            axes_indexes = [axes_tags.index(x) for x in ignore_axes if x in axes_tags]
            return all(
                lyr_coords[i] == active_master_coords[i] 
                for i in axes_indexes 
                if i < len(lyr_coords) and i < len(active_master_coords)
            )
        
        if layer.isBracketLayer():
            current_rules = layer.attributes.get("axisRules", {})
            if current_rules:
                for lyr in glyph.layers:
                    if lyr.isBracketLayer():
                        lyr_rules = lyr.attributes.get("axisRules", {})
                        if lyr_rules == current_rules:
                            layer_ids.add(lyr.layerId)
        else:
            for lyr in glyph.layers:
                if ignore_axes:
                    if lyr.layerId == lyr.associatedMasterId or lyr.isBraceLayer():
                        if match_ignored_axes(lyr):
                            layer_ids.add(lyr.layerId)
                else:
                    if lyr.layerId == lyr.associatedMasterId or lyr.isBraceLayer():
                        layer_ids.add(lyr.layerId)
        
        return list(layer_ids)
    
    @staticmethod
    def check_compatible_angles(glyph, layer_ids: List[str], shape_index: int, node_index: int) -> Tuple[bool, float]:
        """
        Check if angles are compatible across all layers.
        
        Returns:
            Tuple of (is_compatible, max_angle_diff)
        """
        if not glyph.mastersCompatibleForLayerIds_(layer_ids):
            return True, 0.0  # Not compatible, treat as safe
        
        angles = []
        for layer_id in layer_ids:
            layer = glyph.layers[layer_id]
            try:
                current_path = layer.shapes[shape_index]
            except (IndexError, AttributeError):
                continue
            if current_path:
                try:
                    current_node = current_path.nodes[node_index]
                except (IndexError, AttributeError):
                    continue
                if current_node:
                    prev_node, next_node = KinkDetector.get_prev_next_nodes(current_path, node_index)
                    pos1 = (prev_node.position.x, prev_node.position.y)
                    pos2 = (next_node.position.x, next_node.position.y)
                    angles.append(KinkDetector.get_angle(pos1, pos2))
        
        if not angles:
            return True, 0.0
        
        min_angle = min(angles)
        max_angle = max(angles)
        max_diff = max_angle - min_angle
        
        # Compatible if angle diff <= 1.0 degree
        return max_diff <= 1.0, max_diff
    
    @staticmethod
    def check_compatible_proportions(glyph, layer_ids: List[str], shape_index: int, node_index: int, original_hypot: Tuple[float, float]) -> bool:
        """Check if handle proportions are compatible across all layers."""
        if not glyph.mastersCompatibleForLayerIds_(layer_ids):
            return True  # Not compatible, treat as safe
        
        for layer_id in layer_ids:
            layer = glyph.layers[layer_id]
            try:
                current_path = layer.shapes[shape_index]
            except (IndexError, AttributeError):
                continue
            if current_path:
                try:
                    current_node = current_path.nodes[node_index]
                except (IndexError, AttributeError):
                    continue
                if current_node:
                    prev_node, next_node = KinkDetector.get_prev_next_nodes(current_path, node_index)
                    node_pos = (current_node.position.x, current_node.position.y)
                    prev_pos = (prev_node.position.x, prev_node.position.y)
                    next_pos = (next_node.position.x, next_node.position.y)
                    
                    hyp1 = KinkDetector.hypotenuse(node_pos, prev_pos)
                    hyp2 = KinkDetector.hypotenuse(node_pos, next_pos)
                    
                    try:
                        factor = 100 / (hyp1 + hyp2)
                    except ZeroDivisionError:
                        factor = 0
                    try:
                        original_factor = 100 / (original_hypot[0] + original_hypot[1])
                    except ZeroDivisionError:
                        original_factor = 0
                    
                    proportion1 = factor * hyp1
                    proportion2 = original_factor * original_hypot[0]
                    
                    # Compatible if proportion diff <= 0.5%
                    round_error = 0.5
                    if not (proportion1 >= proportion2 - round_error and proportion1 <= proportion2 + round_error):
                        return False
        
        return True
    
    @staticmethod
    def find_potential_kinks(layer, font) -> List[Tuple[float, float, float, str, bool]]:
        """
        Find all smooth nodes that may produce kinks due to incompatible
        angles or proportions across masters.
        
        This is called once when a glyph is selected, not during interpolation.
        
        Args:
            layer: The current GSLayer
            font: The GSFont
            
        Returns:
            List of (x, y, max_angle_diff, node_id, has_ignored_axes) tuples
            for nodes that are potential kinks.
        """
        potential_kinks = []
        
        if not layer or not font:
            return potential_kinks
        
        glyph = layer.parent
        if not glyph:
            return potential_kinks
        
        layer_ids = KinkDetector.get_layer_ids(layer, font)
        
        # Need at least 2 layers to compare
        if len(layer_ids) <= 1:
            return potential_kinks
        
        # Current layer must be in the layer_ids
        if layer.layerId not in layer_ids:
            return potential_kinks
        
        # Masters must be compatible
        if not glyph.mastersCompatibleForLayerIds_(layer_ids):
            return potential_kinks
        
        if layer.countOfPaths() == 0:
            return potential_kinks
        
        # Check for ignored axes
        has_ignored_axes = False
        if "Ignore Kinks Along Axes" in font.customParameters:
            ignore_param = font.customParameters["Ignore Kinks Along Axes"]
            if ignore_param:
                has_ignored_axes = True
        
        for path_index, path in enumerate(layer.shapes):
            # Skip non-path shapes (components)
            if not hasattr(path, 'nodes'):
                continue
            
            nodes = path.nodes
            num_nodes = len(nodes)
            
            for node_index, node in enumerate(nodes):
                # Only check smooth on-curve nodes
                if not node.smooth or node.type == GSOFFCURVE:
                    continue
                
                # Skip first/last nodes on open paths
                if not path.closed:
                    if node_index == 0 or node_index == num_nodes - 1:
                        continue
                
                prev_node, next_node = KinkDetector.get_prev_next_nodes(path, node_index)
                node_pos = (node.position.x, node.position.y)
                prev_pos = (prev_node.position.x, prev_node.position.y)
                next_pos = (next_node.position.x, next_node.position.y)
                
                # Calculate hypotenuses for proportion check
                hyp1 = KinkDetector.hypotenuse(node_pos, prev_pos)
                hyp2 = KinkDetector.hypotenuse(node_pos, next_pos)
                
                # Check angle compatibility
                compatible_angles, max_angle_diff = KinkDetector.check_compatible_angles(
                    glyph, layer_ids, path_index, node_index
                )
                
                # Check proportion compatibility
                compatible_proportions = KinkDetector.check_compatible_proportions(
                    glyph, layer_ids, path_index, node_index, (hyp1, hyp2)
                )
                
                # Node is a potential kink if BOTH angles AND proportions are incompatible
                if not compatible_angles and not compatible_proportions:
                    node_id = f"p{path_index}_n{node_index}"
                    potential_kinks.append((
                        node_pos[0],
                        node_pos[1],
                        max_angle_diff,
                        node_id,
                        has_ignored_axes
                    ))
        
        return potential_kinks
    
    @staticmethod
    def precompute_kinks(
        potential_kinks: List[Tuple],
        glyph_points: Dict[int, Any],
        model: Any,
        axis_min_max: Dict[str, Tuple[float, float]],
        decomposed_layer: Any,
        font: Any,
        cache: 'InterpolationCache'
    ) -> None:
        """
        Precompute max kink severities by sampling interpolations between masters.
        
        Only samples for nodes already identified as potential kinks by find_potential_kinks().
        
        Args:
            potential_kinks: List from find_potential_kinks() - nodes to check
            glyph_points: Dict mapping master index to GlyphCoordinates
            model: VariationModel instance
            axis_min_max: Dict of axis_tag -> (min, max) tuples
            decomposed_layer: Reference decomposed layer for structure
            font: GSFont instance
            cache: InterpolationCache instance to store results
        """
        if not model or not axis_min_max or not potential_kinks:
            return
        
        try:
            if not GlyphInterpolator.validate_points_compatibility(glyph_points):
                return
            
            all_points = list(glyph_points.values())
            axes = list(font.axes)
            masters_list = list(font.masters)
            num_masters = len(masters_list)
            
            if num_masters < 2:
                return
            
            # Build node-to-point mapping once
            node_to_point = GlyphInterpolator.build_node_to_point_map(decomposed_layer)
            
            # Parse potential kink node IDs to get indices
            kink_nodes = []
            for kink_data in potential_kinks:
                _, _, _, node_id, _ = kink_data
                try:
                    parts = node_id.split('_')
                    path_idx = int(parts[0][1:])
                    node_idx = int(parts[1][1:])
                    kink_nodes.append((node_id, path_idx, node_idx))
                except (ValueError, IndexError):
                    continue
            
            if not kink_nodes:
                return
            
            # Generate sample locations between master pairs
            sample_locations = []
            for i in range(num_masters):
                for j in range(i + 1, num_masters):
                    master_a = masters_list[i]
                    master_b = masters_list[j]
                    
                    for t in [0.25, 0.5, 0.75]:
                        sample_loc = {}
                        for ax_idx, axis in enumerate(axes):
                            val_a = master_a.internalAxesValues[ax_idx]
                            val_b = master_b.internalAxesValues[ax_idx]
                            sample_loc[axis.axisTag] = val_a + t * (val_b - val_a)
                        sample_locations.append(sample_loc)
            
            # Sample all locations and compute severities for kink nodes only
            for sample_location in sample_locations:
                normalized_location = {}
                for axis in axes:
                    tag = axis.axisTag
                    normalized_location[tag] = normalizeValue(
                        sample_location[tag],
                        (axis_min_max[tag][0], axis_min_max[tag][0], axis_min_max[tag][1]),
                    )
                
                scalars = model.getScalars(normalized_location)
                if len(scalars) != len(all_points):
                    continue
                
                interpolated_points = model.interpolateFromValuesAndScalars(all_points, scalars)
                interp_list = list(interpolated_points)
                
                # Calculate severity for each potential kink node
                for node_id, path_idx, node_idx in kink_nodes:
                    # Get flat indices
                    flat_idx = node_to_point.get((path_idx, node_idx))
                    if flat_idx is None or flat_idx >= len(interp_list):
                        continue
                    
                    path = decomposed_layer.paths[path_idx]
                    num_nodes = len(path.nodes)
                    prev_node_idx = (node_idx - 1) % num_nodes
                    next_node_idx = (node_idx + 1) % num_nodes
                    
                    prev_flat_idx = node_to_point.get((path_idx, prev_node_idx))
                    next_flat_idx = node_to_point.get((path_idx, next_node_idx))
                    
                    if (prev_flat_idx is None or next_flat_idx is None or
                        prev_flat_idx >= len(interp_list) or next_flat_idx >= len(interp_list)):
                        continue
                    
                    p1 = (interp_list[prev_flat_idx][0], interp_list[prev_flat_idx][1])
                    p2 = (interp_list[flat_idx][0], interp_list[flat_idx][1])
                    p3 = (interp_list[next_flat_idx][0], interp_list[next_flat_idx][1])
                    
                    severity = KinkDetector.calculate_kink_severity(p1, p2, p3)
                    cache.update_kink_severity(node_id, severity)
        
        except Exception as e:
            if DEBUG_INTERPOLATE:
                print(f"PRECOMPUTE KINKS ERROR: {e}")
                import traceback
                traceback.print_exc()


# =============================================================================
# Drawing Helpers
# =============================================================================

class DrawingHelpers:
    """
    Helper methods for common drawing operations in the Reporter.
    
    All methods account for view scale to maintain consistent visual sizes.
    """
    
    @staticmethod
    def draw_circle(
        center: Tuple[float, float],
        size: float,
        fill_color: Optional[Tuple[float, ...]] = None,
        stroke_color: Optional[Tuple[float, ...]] = None,
        stroke_width: float = 1.0,
        scale: float = 1.0
    ) -> Any:
        """
        Draw a circle with optional fill and stroke.
        
        Args:
            center: (x, y) tuple for circle center
            size: Radius in points (will be adjusted for scale)
            fill_color: RGBA tuple for fill, or None
            stroke_color: RGBA tuple for stroke, or None
            stroke_width: Stroke width in points
            scale: View scale factor
            
        Returns:
            NSBezierPath object
        """
        adjusted_size = size / scale
        circle = NSBezierPath.bezierPathWithOvalInRect_((
            (center[0] - adjusted_size, center[1] - adjusted_size),
            (adjusted_size * 2, adjusted_size * 2)
        ))
        if fill_color:
            NSColor.colorWithRed_green_blue_alpha_(*fill_color).set()
            circle.fill()
        if stroke_color:
            NSColor.colorWithRed_green_blue_alpha_(*stroke_color).set()
            circle.setLineWidth_(stroke_width / scale)
            circle.stroke()
        return circle

    @staticmethod
    def draw_star_morph(
        center: Tuple[float, float],
        radius: float,
        inner_ratio: float = 0.45,
        points: int = 16,
        star_factor: float = 1.0,
        fill_color: Optional[Tuple[float, ...]] = None,
        stroke_color: Optional[Tuple[float, ...]] = None,
        stroke_width: float = 1.0,
        scale: float = 1.0
    ) -> Any:
        """
        Draw a starburst shape that interpolates from a circle (star_factor=0)
        to a full star (star_factor=1). Uses alternating outer/inner radii.
        """
        if points < 2:
            return None

        # Clamp inputs to avoid degenerate shapes
        star_factor = max(0.0, min(1.0, star_factor))
        inner_ratio = max(0.05, min(0.95, inner_ratio))

        adjusted_radius = radius / scale
        # Keep outer radius fixed so the shape matches the circle size; spikes grow via inner radius change
        outer_r = adjusted_radius
        inner_r = adjusted_radius * (1.0 - (1.0 - inner_ratio) * star_factor)

        path = NSBezierPath.alloc().init()
        angle_step = (2 * math.pi) / (points * 2)

        for i in range(points * 2):
            angle = i * angle_step
            r = outer_r if i % 2 == 0 else inner_r
            x = center[0] + math.cos(angle) * r
            y = center[1] + math.sin(angle) * r
            if i == 0:
                path.moveToPoint_((x, y))
            else:
                path.lineToPoint_((x, y))

        path.closePath()

        if fill_color:
            NSColor.colorWithRed_green_blue_alpha_(*fill_color).set()
            path.fill()
        if stroke_color:
            NSColor.colorWithRed_green_blue_alpha_(*stroke_color).set()
            path.setLineWidth_(stroke_width / scale)
            path.stroke()
        return path
    
    @staticmethod
    def draw_text(
        text: str,
        position: Tuple[float, float],
        font_size: float = 9.0,
        color: Tuple[float, ...] = (0, 0, 0, 1),
        scale: float = 1.0
    ) -> None:
        """
        Draw text at the specified position.
        
        Args:
            text: String to draw
            position: (x, y) tuple
            font_size: Font size in points
            color: RGBA tuple
            scale: View scale factor
        """
        try:
            adjusted_size = font_size / scale
            textAttrs = {
                NSFontAttributeName: NSFont.systemFontOfSize_(adjusted_size),
                NSForegroundColorAttributeName: NSColor.colorWithRed_green_blue_alpha_(*color)
            }
            nsString = NSString.stringWithString_(text)
            nsString.drawAtPoint_withAttributes_(position, textAttrs)
        except Exception:
            pass
    
    @staticmethod
    def draw_cross(
        center: Tuple[float, float],
        size: float,
        color: Tuple[float, ...],
        stroke_width: float = 1.25,
        scale: float = 1.0
    ) -> None:
        """
        Draw an X-shaped cross mark.
        
        Args:
            center: (x, y) tuple for cross center
            size: Half-size of the cross arms
            color: RGBA tuple
            stroke_width: Line width
            scale: View scale factor
        """
        adjusted_size = size / scale
        NSColor.colorWithRed_green_blue_alpha_(*color).set()
        cross = NSBezierPath.alloc().init()
        cross.moveToPoint_((center[0] - adjusted_size, center[1] + adjusted_size))
        cross.lineToPoint_((center[0] + adjusted_size, center[1] - adjusted_size))
        cross.moveToPoint_((center[0] + adjusted_size, center[1] + adjusted_size))
        cross.lineToPoint_((center[0] - adjusted_size, center[1] - adjusted_size))
        cross.setLineWidth_(stroke_width / scale)
        cross.stroke()
    
    @staticmethod
    def draw_dashed_line(
        start: Tuple[float, float],
        end: Tuple[float, float],
        color: Tuple[float, ...],
        line_width: float = 0.6,
        dash_pattern: Tuple[float, float] = (1.0, 1.5),
        scale: float = 1.0,
        use_dash: bool = True
    ) -> None:
        """
        Draw a line between two points (optionally dashed).
        
        Args:
            start: (x, y) tuple
            end: (x, y) tuple
            color: RGBA tuple
            line_width: Line width
            dash_pattern: Tuple of (dash_length, gap_length)
            scale: View scale factor
            use_dash: If True, draw dashed line; if False, draw solid line
        """
        NSColor.colorWithRed_green_blue_alpha_(*color).set()
        line = NSBezierPath.alloc().init()
        line.moveToPoint_(start)
        line.lineToPoint_(end)
        line.setLineWidth_(line_width / scale)
        if use_dash:
            line.setLineDash_count_phase_(
                [dash_pattern[0] / scale, dash_pattern[1] / scale], 2, 0.0
            )
        line.stroke()
    
    @staticmethod
    def draw_kink_indicator(
        center: Tuple[float, float],
        current_severity: float,
        max_severity: float,
        angle_deg: float,
        scale: float = 1.0
    ) -> None:
        """
        Draw a kink indicator circle with severity-based styling.
        
        The circle scales based on current severity relative to max severity
        for THIS specific node, so each node reaches its biggest size when
        it's at its personal maximum kink.
        
        Args:
            center: (x, y) tuple
            current_severity: Current kink severity (0-200)
            max_severity: Maximum severity recorded for this specific node
            angle_deg: Current kink angle in degrees (0 = smooth, higher = more kinked)
            scale: View scale factor
        """
        # Per-node peak severity for scaling the star factor (never shared across nodes)
        node_peak = max(max_severity, 0.0001)
        severity_norm = min(1.0, current_severity / node_peak)
        # Gate star morph with angle: 0 at 0.7°, fully allowed by 0.8°; then scale by severity
        angle_gate = min(1.0, max(0.0, (angle_deg - 0.7) / 0.1))
        star_factor = severity_norm * angle_gate
        
        # Fixed size for the base circle; only the star shape indicates intensity
        base_size = 9.0
        
        # Keep alpha fixed so opacity does not encode intensity
        alpha = 0.7
        
        # Use configurable kink indicator color
        kink_indicator_color = InterpolateConfig.get_kink_indicator_color()
        kink_color = (kink_indicator_color[0], kink_indicator_color[1], kink_indicator_color[2])
        
        # Keep a true circle until angle crosses 0.7°; then morph toward star by severity
        if angle_gate <= 0.0:
            DrawingHelpers.draw_circle(
                center, base_size,
                fill_color=(kink_color[0], kink_color[1], kink_color[2], alpha * 0.15),
                stroke_color=(kink_color[0], kink_color[1], kink_color[2], alpha),
                stroke_width=1.5,
                scale=scale
            )
        else:
            DrawingHelpers.draw_star_morph(
                center=center,
                radius=base_size,
                inner_ratio=0.45,
                points=16,
                star_factor=star_factor,
                fill_color=(kink_color[0], kink_color[1], kink_color[2], alpha * 0.15),
                stroke_color=(kink_color[0], kink_color[1], kink_color[2], alpha),
                stroke_width=1.5,
                scale=scale
            )
        
        # Draw angle label only when kink angle > 0.7 degrees
        if angle_deg > 0.7:
            text_x = center[0] + (base_size + 2.0) / scale
            text_y = center[1] + base_size / (2 * scale)
            DrawingHelpers.draw_text(
                f"{angle_deg:.1f}°",
                (text_x, text_y),
                font_size=9.0,
                color=(kink_color[0], kink_color[1], kink_color[2], alpha),
                scale=scale
            )
        
        # Draw inner circle when at or near this node's peak severity (within 10%)
        # Use smaller inner circle (0.35) so it's clearly visible inside the outer one
        if max_severity > 0.001 and current_severity >= max_severity * 0.90:
            # Inner color is a lighter version of the kink indicator color
            inner_color = (
                min(1.0, kink_color[0] + 0.4),
                min(1.0, kink_color[1] + 0.4),
                min(1.0, kink_color[2] + 0.4)
            )
            DrawingHelpers.draw_circle(
                center, base_size * 0.5,
                stroke_color=(inner_color[0], inner_color[1], inner_color[2], alpha),
                stroke_width=1.5,
                scale=scale
            )

    
    @staticmethod
    def draw_tooltip(
        position: Tuple[float, float],
        lines: List[str],
        scale: float = 1.0,
        bg_color: Tuple[float, ...] = (0.1, 0.1, 0.1, 0.85),
        text_color: Tuple[float, ...] = (1, 1, 1, 1)
    ) -> None:
        """
        Draw a tooltip with multiple lines of text.
        
        Args:
            position: (x, y) tuple for top-left corner
            lines: List of strings to display
            scale: View scale factor
            bg_color: RGBA tuple for background
            text_color: RGBA tuple for text
        """
        if not lines:
            return
        
        fontSize = 11.0 / scale
        lineHeight = fontSize * 1.3
        padding = 4 / scale
        textWidth = 100 / scale
        textHeight = lineHeight * len(lines)
        
        x, y = position
        bgRect = NSMakeRect(
            x - padding,
            y - padding,
            textWidth + 2 * padding,
            textHeight + 2 * padding
        )
        
        # Semi-transparent background
        NSColor.colorWithWhite_alpha_(bg_color[0], bg_color[3]).set()
        bgPath = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bgRect, 4 / scale, 4 / scale
        )
        bgPath.fill()
        
        # Draw text lines
        font = NSFont.systemFontOfSize_(fontSize)
        textNSColor = NSColor.colorWithRed_green_blue_alpha_(*text_color)
        
        for i, line in enumerate(lines):
            lineY = y + textHeight - (i + 1) * lineHeight + 2 / scale
            attrs = NSMutableDictionary.alloc().init()
            attrs[NSFontAttributeName] = font
            attrs[NSForegroundColorAttributeName] = textNSColor
            attrString = NSAttributedString.alloc().initWithString_attributes_(line, attrs)
            attrString.drawAtPoint_((x, lineY))
    
    @staticmethod
    def build_bubble_path(
        rect: Any,
        corner_radius: float,
        triangle_tip: Tuple[float, float],
        triangle_width: float,
        triangle_height: float
    ) -> Any:
        """
        Build a rounded rectangle path with a triangular pointer.
        
        Args:
            rect: NSRect for the bubble body
            corner_radius: Radius for rounded corners
            triangle_tip: (x, y) tuple for the triangle tip position
            triangle_width: Width of triangle base
            triangle_height: Height of triangle
            
        Returns:
            NSBezierPath with the complete bubble shape
        """
        path = NSBezierPath.alloc().init()
        
        x, y = rect.origin.x, rect.origin.y
        w, h = rect.size.width, rect.size.height
        r = corner_radius
        
        # Triangle positioning
        triangleTipX, triangleTipY = triangle_tip
        triangleBaseY = triangleTipY + triangle_height
        triangleLeft = triangleTipX - triangle_width / 2
        triangleRight = triangleTipX + triangle_width / 2
        
        # Bottom edge: left to right
        path.moveToPoint_((x + r, y))
        path.lineToPoint_((x + w - r, y))
        
        # Bottom-right corner
        path.curveToPoint_controlPoint1_controlPoint2_(
            (x + w, y + r), (x + w - r * 0.45, y), (x + w, y + r * 0.45)
        )
        
        # Right edge
        path.lineToPoint_((x + w, y + h - r))
        
        # Top-right corner
        path.curveToPoint_controlPoint1_controlPoint2_(
            (x + w - r, y + h), (x + w, y + h - r * 0.45), (x + w - r * 0.45, y + h)
        )
        
        # Top edge to triangle
        path.lineToPoint_((triangleRight, y + h))
        
        # Triangle
        path.lineToPoint_((triangleTipX, triangleBaseY))
        path.lineToPoint_((triangleLeft, y + h))
        
        # Top edge from triangle
        path.lineToPoint_((x + r, y + h))
        
        # Top-left corner
        path.curveToPoint_controlPoint1_controlPoint2_(
            (x, y + h - r), (x + r * 0.45, y + h), (x, y + h - r * 0.45)
        )
        
        # Left edge
        path.lineToPoint_((x, y + r))
        
        # Bottom-left corner
        path.curveToPoint_controlPoint1_controlPoint2_(
            (x + r, y), (x, y + r * 0.45), (x + r * 0.45, y)
        )
        
        path.closePath()
        return path
    
    @staticmethod
    def draw_bubble_with_shadow(
        path: Any,
        fill_color: Tuple[float, ...] = (1, 1, 1, 1),
        stroke_color: Optional[Tuple[float, ...]] = None,
        stroke_width: float = 1.0,
        shadow_offset: Tuple[float, float] = (0, -4),
        shadow_blur: float = 10.0,
        shadow_alpha: float = 0.3
    ) -> None:
        """
        Draw a path with drop shadow.
        
        Args:
            path: NSBezierPath to draw
            fill_color: RGBA tuple for fill
            stroke_color: RGBA tuple for stroke, or None
            stroke_width: Width of stroke
            shadow_offset: (dx, dy) tuple for shadow offset
            shadow_blur: Shadow blur radius
            shadow_alpha: Shadow opacity
        """
        # Save graphics state for shadow
        NSGraphicsContext.currentContext().saveGraphicsState()
        
        # Apply drop shadow
        shadow = NSShadow.alloc().init()
        shadow.setShadowOffset_(shadow_offset)
        shadow.setShadowBlurRadius_(shadow_blur)
        shadow.setShadowColor_(NSColor.colorWithWhite_alpha_(0.0, shadow_alpha))
        shadow.set()
        
        # Fill
        NSColor.colorWithRed_green_blue_alpha_(*fill_color).set()
        path.fill()
        
        # Restore graphics state
        NSGraphicsContext.currentContext().restoreGraphicsState()
        
        # Stroke if specified
        if stroke_color:
            NSColor.colorWithRed_green_blue_alpha_(*stroke_color).set()
            path.setLineWidth_(stroke_width)
            path.stroke()


# Helper function to check if a layer is a newline control layer
def is_newline_layer(layer):
    """Check if a layer is a newline (GSControlLayer with char code 10)"""
    if layer is None:
        return True
    try:
        className = layer.className()
        if 'ControlLayer' in className:
            if hasattr(layer, 'parent') and layer.parent:
                unicodeVal = layer.parent.unicodeChar()
                # Only newline (10) and carriage return (13) are line separators
                return unicodeVal == 10 or unicodeVal == 13
            # Control layer without parent - check if it's a newline by other means
            # Don't assume it's a separator - it might be a space or other character
            return False
    except Exception:
        pass
    return False


# Custom NSView for drawing interpolated glyphs preview
class InterpolatePreviewView(NSView):
    
    def initWithFrame_(self, frame):
        self = objc.super(InterpolatePreviewView, self).initWithFrame_(frame)
        if self:
            self.state = None
            self.glyphPaths = []  # List of (path, width, isSelected) tuples
            self.selectedIndex = 0
        return self
    
    def isFlipped(self):
        return False
    
    def setInterpolateState_(self, state):
        self.state = state
    
    def drawRect_(self, rect):
        # Fill background with white
        NSColor.whiteColor().set()
        NSRectFill(rect)
        
        # Safety check - don't draw if we're being closed
        if not self.glyphPaths or self.state is None:
            return
        
        try:
            bounds = self.bounds()
            viewWidth = bounds.size.width
            viewHeight = bounds.size.height
            
            # Calculate total width including kerning
            # glyphPaths is now list of (path, width, leftKerning, isSelected)
            totalWidth = sum(width + leftKerning for path, width, leftKerning, isSelected in self.glyphPaths)
            if totalWidth == 0:
                return
            
            # Get font metrics for proper vertical scaling
            upm = 1000
            descender = -250  # Default descender
            ascender = 750    # Default ascender
            if Glyphs.font:
                upm = Glyphs.font.upm
                # Get descender and ascender from first master
                if Glyphs.font.masters:
                    master = Glyphs.font.masters[0]
                    descender = master.descender if hasattr(master, 'descender') else -250
                    ascender = master.ascender if hasattr(master, 'ascender') else 750
            
            # Calculate the full vertical extent of glyphs (ascender to descender)
            fullHeight = ascender - descender
            
            # Add padding (10% top and bottom)
            paddingRatio = 0.1
            availableHeight = viewHeight * (1 - 2 * paddingRatio)
            
            # Scale based on window height to fit the full glyph height
            scale = availableHeight / fullHeight
            
            # Find position of selected glyph to center it horizontally
            selectedX = 0
            for i, (path, width, leftKerning, isSelected) in enumerate(self.glyphPaths):
                if i == self.selectedIndex:
                    selectedX += (leftKerning * scale) + (width * scale) / 2
                    break
                selectedX += (width + leftKerning) * scale
            
            # Calculate starting X to center selected glyph
            startX = (viewWidth / 2) - selectedX
            
            # Baseline position: account for descender so glyphs are vertically centered
            # Position baseline so that (ascender + descender)/2 is at view center
            baseline = (viewHeight / 2) - (((ascender + descender) / 2) * scale)
            
            # Draw each glyph
            NSColor.blackColor().set()
            currentX = startX
            
            for path, width, leftKerning, isSelected in self.glyphPaths:
                # Apply left kerning (moves this glyph closer/further from previous)
                currentX += leftKerning * scale
                
                if path:
                    transform = NSAffineTransform.transform()
                    transform.translateXBy_yBy_(currentX, baseline)
                    transform.scaleBy_(scale)
                    
                    transformedPath = path.copy()
                    transformedPath.transformUsingAffineTransform_(transform)
                    transformedPath.fill()
                
                currentX += width * scale
        except Exception:
            pass  # Silently handle drawing errors during close
    
    def updateWithGlyphs_(self, glyphData):
        """Update with list of (bezierPath, width, leftKerning, isSelected) tuples"""
        self.glyphPaths = glyphData
        # Find selected index
        for i, (path, width, leftKerning, isSelected) in enumerate(glyphData):
            if isSelected:
                self.selectedIndex = i
                break
        self.setNeedsDisplay_(True)


# =============================================================================
# Shared State Class
# =============================================================================

class InterpolateState:
    """
    Shared state class to coordinate between the three plugin classes.
    
    NOTE: This is a per-instance class, NOT a singleton.
    Each palette instance creates its own InterpolateState to ensure
    font-specific data doesn't leak between different open fonts.
    
    Uses the centralized InterpolationCache and GlyphInterpolator for
    efficient caching and interpolation operations.
    """
    
    # Class-level registry to track all instances for the Reporter to find
    _instances = {}  # Maps font signature -> state instance
    
    # Class-level flag: is the Reporter currently active in any tab?
    # Set to True when Reporter's background() is called, used by Palette to skip work when inactive
    _reporter_active = False
    _reporter_last_active_time = 0  # Timestamp of last Reporter activity
    
    def __init__(self):
        # Interpolation state
        self.current_location = {}
        self.glyph_points = {}
        self.model = None
        self.master_scalars = []
        self.current_glyph = None
        self.axis_min_max = {}
        
        # Display data (computed by update_glyph)
        self.displaypath = None
        self.interp_nodes = []
        self.interp_handles = []
        self.interpolated_anchors = {}
        self.interpolated_width = 0
        
        # Font reference - set when state is registered with a font
        self._font = None

        # Track model construction to know if we must rebuild when master order changes
        self._model_master_ids = []
        
        # Centralized cache manager
        self._cache = InterpolationCache()
        
        # Track current font to detect font switches
        self._current_font_id = None
        
        # Track axis count for UI rebuild detection
        self._last_axis_count = 0
        
        # Preview update throttling
        self._last_update_time = 0
        
        # Kink tracking
        self.potential_kinks = []  # List of (path_idx, node_idx, node_id, max_angle_diff) from detection
        self.interpolated_kinks = []  # List of (x, y, angle_diff, max_angle_diff, angle_deg) for display

        # Axis animation state (per axis tag)
        # {'wght': {'is_playing': False, 'direction': 1, 'speed': 1.0,
        #           'last_time': None, 'paused_until': None, 'resume_after_scrub': False}}
        self.axis_animation = {}
        
        # UI toggles (persisted in NSUserDefaults - shared across all fonts)
        defaults = NSUserDefaults.standardUserDefaults()
        self.show_preview = defaults.boolForKey_(KEY + ".show_preview") if defaults.objectForKey_(KEY + ".show_preview") is not None else True
        self.show_nodes = defaults.boolForKey_(KEY + ".show_nodes") if defaults.objectForKey_(KEY + ".show_nodes") is not None else False
        self.show_fill = defaults.boolForKey_(KEY + ".show_fill") if defaults.objectForKey_(KEY + ".show_fill") is not None else True
        self.show_anchors = defaults.boolForKey_(KEY + ".show_anchors") if defaults.objectForKey_(KEY + ".show_anchors") is not None else False
        
        # Ensure at least one visualization is active (fill or nodes)
        if not self.show_fill and not self.show_nodes:
            self.show_fill = True
        
        # Preview window
        self.preview_window = None
        self.preview_view = None
        self.show_preview_window = False
        
        # Compare line feature - bubble overlay for interpolation preview
        self.show_compare_line = defaults.boolForKey_(KEY + ".show_compare_line") if defaults.objectForKey_(KEY + ".show_compare_line") is not None else False

        # Follow selected master position once on master switch
        self.follow_masters = defaults.boolForKey_(KEY + ".follow_masters") if defaults.objectForKey_(KEY + ".follow_masters") is not None else False
        
        # Bubble scale (1.0 = same size as edit view, 0.5 = 50%, 2.0 = 200%)
        self.bubble_scale = defaults.floatForKey_(KEY + ".bubble_scale") if defaults.objectForKey_(KEY + ".bubble_scale") is not None else 1.0
        if self.bubble_scale <= 0:
            self.bubble_scale = 1.0
        
        # Show kinks toggle
        self.show_kinks = defaults.boolForKey_(KEY + ".show_kinks") if defaults.objectForKey_(KEY + ".show_kinks") is not None else False
        
        # Dim the main layer to make interpolation preview more visible
        self.dim_layer = defaults.boolForKey_(KEY + ".dim_layer") if defaults.objectForKey_(KEY + ".dim_layer") is not None else False
        
        # Spacebar preview shows interpolated glyphs instead of current master
        self.space_preview_interpolated = defaults.boolForKey_(KEY + ".space_preview_interpolated") if defaults.objectForKey_(KEY + ".space_preview_interpolated") is not None else False
        
        # Tool axis mapping for mouse drag control
        # These store the axis tag (e.g., "wght", "wdth") for horizontal/vertical mouse movement
        # None means "no axis" - that direction is disabled
        self.tool_horizontal_axis = None  # Axis controlled by horizontal mouse movement
        self.tool_vertical_axis = None    # Axis controlled by vertical mouse movement
        
        # Palette reference - set when palette registers with this state
        # Used by the tool to sync palette sliders after drag
        self._palette = None

        # Track last seen master to detect master switches
        self._last_master_id = None
    
    # =========================================================================
    # Class Methods for Font/State Management
    # =========================================================================
    
    @classmethod
    def get_state_for_font(cls, font):
        """Get or create a state instance for a specific font.
        Used by the Reporter plugin to find the correct state for the current font."""
        if font is None:
            return None
        font_sig = cls._get_font_signature_static(font)
        return cls._instances.get(font_sig)
    
    @classmethod
    def register_state(cls, font, state):
        """Register a state instance for a font signature."""
        if font is None:
            return
        font_sig = cls._get_font_signature_static(font)
        cls._instances[font_sig] = state
        # Store font reference in state so update_glyph can use the correct font
        state._font = font
        debug_log(f"InterpolateState: Registered state for font '{font_sig}'")
    
    @classmethod
    def unregister_state(cls, font):
        """Unregister a state instance for a font signature."""
        if font is None:
            return
        font_sig = cls._get_font_signature_static(font)
        if font_sig in cls._instances:
            del cls._instances[font_sig]
            debug_log(f"InterpolateState: Unregistered state for font '{font_sig}'")
    
    @staticmethod
    def _get_font_signature_static(font):
        """Get a stable signature for a font (static method for class-level use)."""
        if font is None:
            return None
        file_path = getattr(font, 'filePath', None) or getattr(font, 'filepath', None) or 'untitled'
        family_name = getattr(font, 'familyName', 'Unknown')
        axis_count = len(font.axes) if font.axes else 0
        return f"{file_path}|{family_name}|{axis_count}"
    
    def _get_font_signature(self, font):
        """Get a stable signature for a font that won't change like id(font) does."""
        return self._get_font_signature_static(font)
    
    # =========================================================================
    # Preference Management
    # =========================================================================
    
    def save_defaults(self):
        """Save UI toggle states to NSUserDefaults."""
        defaults = NSUserDefaults.standardUserDefaults()
        defaults.setBool_forKey_(self.show_preview, KEY + ".show_preview")
        defaults.setBool_forKey_(self.show_nodes, KEY + ".show_nodes")
        defaults.setBool_forKey_(self.show_fill, KEY + ".show_fill")
        defaults.setBool_forKey_(self.show_anchors, KEY + ".show_anchors")
        defaults.setBool_forKey_(self.show_compare_line, KEY + ".show_compare_line")
        defaults.setBool_forKey_(self.follow_masters, KEY + ".follow_masters")
        defaults.setFloat_forKey_(self.bubble_scale, KEY + ".bubble_scale")
        defaults.setBool_forKey_(self.show_kinks, KEY + ".show_kinks")
        defaults.setBool_forKey_(self.dim_layer, KEY + ".dim_layer")
        defaults.setBool_forKey_(self.space_preview_interpolated, KEY + ".space_preview_interpolated")
        try:
            defaults.synchronize()
        except Exception:
            pass
    
    # =========================================================================
    # Cache Management (delegating to InterpolationCache)
    # =========================================================================
    
    def clear_caches(self, clear_kinks=False):
        """Clear interpolation caches.
        
        Args:
            clear_kinks: If True, also clear kink detection.
                         Should be True when switching to a different glyph,
                         False when just updating interpolation position.
        """
        self._cache.invalidate_glyph(clear_kinks=clear_kinks)
        self.glyph_points = {}
        if clear_kinks:
            self.potential_kinks = []
            self.interpolated_kinks = []

    def clear_all_caches(self):
        """Clear ALL caches - call when switching fonts or turning off plugin."""
        self._cache.invalidate_all()
        
        # Clear display data
        self.glyph_points = {}
        self.displaypath = None
        self.interp_nodes = []
        self.interp_handles = []
        self.interpolated_anchors = {}
        self.potential_kinks = []
        self.interpolated_kinks = []
        
        # Clear model data (font-specific)
        self.model = None
        self.master_scalars = []
        self.current_glyph = None
        self.axis_min_max = {}
        self.current_location = {}
        self._last_master_id = None

    # =========================================================================
    # Axis Value Conversion Helpers
    # =========================================================================
    
    def get_actual_value(self, axis_tag: str) -> float:
        """
        Convert normalized axis value (0-1) to actual design space value.
        
        Args:
            axis_tag: The axis tag (e.g., 'wght', 'wdth')
            
        Returns:
            The actual axis value in design space units, or 0 if axis not found.
        """
        if axis_tag not in self.current_location or axis_tag not in self.axis_min_max:
            return 0.0
        norm_val = self.current_location[axis_tag]
        min_val, max_val = self.axis_min_max[axis_tag]
        return min_val + norm_val * (max_val - min_val)
    
    def set_from_actual_value(self, axis_tag: str, actual_value: float) -> bool:
        """
        Set axis location from actual design space value.
        
        Args:
            axis_tag: The axis tag (e.g., 'wght', 'wdth')
            actual_value: The actual axis value in design space units
            
        Returns:
            True if value was set successfully, False if axis not found.
        """
        if axis_tag not in self.axis_min_max:
            return False
        min_val, max_val = self.axis_min_max[axis_tag]
        if max_val == min_val:
            norm_val = 0.0
        else:
            norm_val = (actual_value - min_val) / (max_val - min_val)
        # Clamp to valid range
        norm_val = max(0.0, min(1.0, norm_val))
        self.current_location[axis_tag] = norm_val
        return True

    def set_location_from_master(self, master, axes) -> bool:
        """Set current location to the selected master's axis values."""
        if not master or not axes or not self.axis_min_max:
            return False
        try:
            normalized_location = {}
            for ix, axis in enumerate(axes):
                axis_tag = axis.axisTag
                if axis_tag not in self.axis_min_max:
                    continue
                min_val, max_val = self.axis_min_max[axis_tag]
                normalized_location[axis_tag] = normalizeValue(
                    master.internalAxesValues[ix],
                    (min_val, min_val, max_val),
                )

            if not normalized_location:
                return False

            self.current_location = normalized_location
            if self.model:
                self.master_scalars = self.model.getMasterScalars(self.current_location)
            self.clear_caches(clear_kinks=False)
            self.update_glyph()
            return True
        except Exception as e:
            debug_log(f"set_location_from_master error: {e}")
            return False

    def ensure_axis_animation(self, axis_tag: str) -> Dict[str, Any]:
        """
        Ensure an animation state entry exists for the given axis.
        """
        if axis_tag not in self.axis_animation:
            self.axis_animation[axis_tag] = {
                "is_playing": False,
                "direction": 1,
                "speed": 1.0,
                "last_time": None,
                "paused_until": None,
                "resume_after_scrub": False,
            }
        return self.axis_animation[axis_tag]

    def apply_axis_updates(self, updates: Dict[str, float]) -> None:
        """
        Apply actual-value updates for multiple axes, then recompute scalars and glyph.
        """
        if not updates:
            return

        for axis_tag, actual_value in updates.items():
            self.set_from_actual_value(axis_tag, actual_value)

        if self.model:
            self.master_scalars = self.model.getMasterScalars(self.current_location)
        self.update_glyph()

    # =========================================================================
    # Font Bounds
    # =========================================================================
    
    def get_font_bounds(self, font):
        """Get cached font-wide bounds for performance."""
        font_sig = self._get_font_signature(font)
        
        # Check cache first
        cached = self._cache.get_font_bounds(font_sig)
        if cached is not None:
            return cached
        
        # Calculate font-wide bounds
        fontMinY = -250
        fontMaxY = 750
        
        if font.masters:
            master = font.masters[0]
            fontMinY = getattr(master, 'descender', -250)
            fontMaxY = getattr(master, 'ascender', 750)
        
        for glyph in font.glyphs:
            for layer in glyph.layers:
                if layer.bounds:
                    bounds = layer.bounds
                    if bounds.size.width > 0 and bounds.size.height > 0:
                        layerMinY = bounds.origin.y
                        layerMaxY = bounds.origin.y + bounds.size.height
                        fontMinY = min(fontMinY, layerMinY)
                        fontMaxY = max(fontMaxY, layerMaxY)
        
        # Cache and return
        result = (fontMinY, fontMaxY)
        self._cache.set_font_bounds(font_sig, result)
        return result
    
    # =========================================================================
    # Model Building
    # =========================================================================
    
    def build_model(self):
        """Build variation model for interpolation."""
        # Use the font associated with this state
        font = self._font if self._font else Glyphs.font
        if not font:
            debug_log("build_model(): No font, skipping")
            return

        # Rebuild if no model or masters changed (order/ids)
        current_master_ids = [m.id for m in font.masters] if font.masters else []
        if self.model and self._model_master_ids == current_master_ids:
            debug_log("build_model(): Already have model with matching masters, skipping")
            return

        # Guard: need axis_min_max to be populated
        if not self.axis_min_max:
            debug_log("build_model(): No axis_min_max, skipping")
            return

        layers = font.selectedLayers
        if not layers:
            debug_log("build_model(): No selectedLayers, skipping")
            return
        layer = layers[0]

        debug_log(f"build_model(): Building model for font with axes {list(self.axis_min_max.keys())}")
        
        # Cache font properties to avoid repeated lookups
        masters = font.masters
        axes = font.axes
        axis_min_max = self.axis_min_max
        
        locations = []
        # Use font.masters order to keep VariationModel locations aligned with glyph_points ordering.
        for master in masters:
            normalized_location = {}
            for ix, axis in enumerate(axes):
                axis_tag = axis.axisTag
                if axis_tag not in axis_min_max:
                    continue
                min_val = axis_min_max[axis_tag][0]
                max_val = axis_min_max[axis_tag][1]
                normalized_location[axis_tag] = normalizeValue(
                    master.internalAxesValues[ix],
                    (min_val, min_val, max_val),
                )
            locations.append(normalized_location)
        
        self.model = VariationModel(locations)
        debug_log(f"build_model(): Model built successfully with {len(locations)} locations")
        self._model_master_ids = current_master_ids
        
        # Initialize master_scalars with current_location if available
        if self.current_location:
            self.master_scalars = self.model.getMasterScalars(self.current_location)
            debug_log(f"build_model(): Initialized master_scalars with {len(self.master_scalars)} values")
        else:
            # If no current_location yet, use the first master's location as default
            if locations:
                self.current_location = dict(locations[0])
                self.master_scalars = self.model.getMasterScalars(self.current_location)
                debug_log(f"build_model(): No current_location, using first master location. master_scalars has {len(self.master_scalars)} values")
    
    # =========================================================================
    # Glyph Interpolation (using GlyphInterpolator)
    # =========================================================================
    
    def update_glyph(self):
        """Update the interpolated glyph display path and related data."""
        # Get font reference
        font = self._font if self._font else Glyphs.font
        if not font:
            debug_log("update_glyph(): No font, skipping")
            return
        if not font.selectedLayers:
            debug_log("update_glyph(): No selectedLayers, skipping")
            return
        if not self.master_scalars:
            debug_log(f"update_glyph(): No master_scalars, skipping")
            return
        if not self.model:
            debug_log("update_glyph(): No model, skipping")
            return
        
        # Debug: show current_location values being used for interpolation
        if DEBUG_INTERPOLATE and self.current_location and self.axis_min_max:
            loc_info = [f"{tag}={self.get_actual_value(tag):.1f}" for tag in self.current_location if tag in self.axis_min_max]
            print(f"[Interpol] update_glyph(): Using location: {', '.join(loc_info)}")
        
        try:
            current_layer = font.selectedLayers[0]
            if not current_layer or not current_layer.parent:
                return
            current_layer_id = current_layer.layerId
            
            # Get decomposed layer for first master (consistent reference structure)
            glyph = current_layer.parent
            first_master_layer = glyph.layers[font.masters[0].id]
            current_decomposed = GlyphInterpolator.get_decomposed_layer(first_master_layer)
            
            # Rebuild the .glyph_points array using master layers (indexed by master index)
            # Also collect widths for interpolation
            layer_widths = {}
            for ix, master in enumerate(font.masters):
                layer = glyph.layers[master.id]
                if not layer:
                    continue
                if ix not in self.glyph_points or layer.layerId == current_layer_id:
                    decomposed = GlyphInterpolator.get_decomposed_layer(layer)
                    self.glyph_points[ix] = GlyphInterpolator.extract_points_from_layer(decomposed)
                
                # Store the layer width for interpolation
                layer_widths[ix] = layer.width

            # Validate point compatibility
            if not GlyphInterpolator.validate_points_compatibility(self.glyph_points):
                return
            
            all_points = list(self.glyph_points.values())
            if len(self.master_scalars) != len(all_points):
                return
            
            # Interpolate points
            interpolated_points = GlyphInterpolator.interpolate_points(
                self.glyph_points, self.model, self.master_scalars
            )
            if interpolated_points is None:
                return
            
            # Interpolate the width
            all_widths = [layer_widths.get(i, current_layer.width) for i in range(len(self.master_scalars))]
            self.interpolated_width = GlyphInterpolator.interpolate_width(all_widths, self.master_scalars)

            # Build bezier path with node tracking
            self.displaypath, self.interp_nodes, self.interp_handles = \
                GlyphInterpolator.build_bezier_path(current_decomposed, interpolated_points)
            
            # Build kink display data from potential_kinks
            # Map each potential kink to its interpolated position and calculate current severity
            self.interpolated_kinks = []
            if self.show_kinks and self.potential_kinks:
                # Build mapping from (path_idx, node_idx) to flat point index
                node_to_point = GlyphInterpolator.build_node_to_point_map(current_decomposed)
                
                for kink_data in self.potential_kinks:
                    orig_x, orig_y, max_angle_diff, node_id, has_ignored_axes = kink_data
                    
                    # Parse the node_id to get path and node indices
                    try:
                        parts = node_id.split('_')
                        path_idx = int(parts[0][1:])  # "p0" -> 0
                        node_idx = int(parts[1][1:])  # "n5" -> 5
                    except (ValueError, IndexError):
                        continue
                    
                    # Look up flat index using the node-to-point map
                    flat_idx = node_to_point.get((path_idx, node_idx))
                    if flat_idx is None or flat_idx >= len(interpolated_points):
                        continue
                    
                    # Get interpolated position for this node
                    try:
                        point_data = interpolated_points[flat_idx]
                        interp_x, interp_y = point_data[0], point_data[1]
                    except Exception:
                        continue
                    
                    # Get prev and next node indices for severity calculation
                    path = current_decomposed.paths[path_idx]
                    num_nodes = len(path.nodes)
                    prev_node_idx = (node_idx - 1) % num_nodes
                    next_node_idx = (node_idx + 1) % num_nodes
                    
                    # Look up prev and next flat indices
                    prev_flat_idx = node_to_point.get((path_idx, prev_node_idx))
                    next_flat_idx = node_to_point.get((path_idx, next_node_idx))
                    
                    # Calculate current kink severity using dot product method
                    current_severity = 0.0
                    if (prev_flat_idx is not None and next_flat_idx is not None and
                        prev_flat_idx < len(interpolated_points) and 
                        next_flat_idx < len(interpolated_points)):
                        prev_point = interpolated_points[prev_flat_idx]
                        next_point = interpolated_points[next_flat_idx]
                        
                        p1 = (prev_point[0], prev_point[1])
                        p2 = (interp_x, interp_y)
                        p3 = (next_point[0], next_point[1])
                        
                        # Calculate severity using dot product (0 = smooth, higher = more kinked)
                        current_severity = KinkDetector.calculate_kink_severity(p1, p2, p3)
                    
                    # Update max severity tracking for this node
                    self._cache.update_kink_severity(node_id, current_severity)
                    
                    # Get the max severity recorded for this specific node
                    max_severity = self._cache.get_kink_severity(node_id)
                    
                    # Convert severity to kink angle for display (0° = smooth, higher = kinked)
                    current_angle_deg = KinkDetector.severity_to_degrees(current_severity)
                    
                    self.interpolated_kinks.append((
                        interp_x,
                        interp_y,
                        current_severity,    # current severity (0-200 scale)
                        max_severity,        # max severity for this specific node
                        current_angle_deg    # kink angle in degrees for display
                    ))
            
            # Interpolate anchors
            glyph = current_layer.parent
            self.interpolated_anchors = GlyphInterpolator.interpolate_anchors(
                glyph, font, self.master_scalars
            )
            
        except Exception:
            pass  # Silently handle glyph update errors
        
        Glyphs.redraw()
        
        # Schedule throttled preview window update
        self.schedule_preview_update()
    
    # =========================================================================
    # Kink Detection (delegating to KinkDetector)
    # =========================================================================
    
    def _detect_potential_kinks(self):
        """
        Detect potential kink nodes by comparing angles and proportions across masters.
        
        This is called once when a glyph is selected. The results are stored in
        self.potential_kinks and used during update_glyph() to determine which
        nodes to display as kinks.
        
        This approach matches showAngleProportionKink.py:
        - A node is a kink if angles differ by >1° across masters
        - AND handle proportions differ by >0.5% across masters
        
        Also precomputes max severities by sampling interpolations between masters.
        """
        font = self._font if self._font else Glyphs.font
        if not font or not font.selectedLayers:
            self.potential_kinks = []
            return
        
        try:
            current_layer = font.selectedLayers[0]
            if not current_layer or not current_layer.parent:
                self.potential_kinks = []
                return
            
            # Delegate to KinkDetector to find potential kinks
            self.potential_kinks = KinkDetector.find_potential_kinks(current_layer, font)
            
            # Precompute max severities for these nodes by sampling interpolations
            if self.potential_kinks and self.model and self.axis_min_max:
                current_decomposed = GlyphInterpolator.get_decomposed_layer(current_layer)
                
                # Ensure glyph_points is populated
                glyph = current_layer.parent
                masters = font.masters
                for ix, master in enumerate(masters):
                    if ix not in self.glyph_points:
                        layer = glyph.layers[master.id]
                        if layer:
                            decomposed = GlyphInterpolator.get_decomposed_layer(layer)
                            self.glyph_points[ix] = GlyphInterpolator.extract_points_from_layer(decomposed)
                
                # Precompute max severities
                KinkDetector.precompute_kinks(
                    self.potential_kinks,
                    self.glyph_points,
                    self.model,
                    self.axis_min_max,
                    current_decomposed,
                    font,
                    self._cache
                )
            
        except Exception as e:
            if DEBUG_INTERPOLATE:
                print(f"DETECT KINKS ERROR: {e}")
                import traceback
                traceback.print_exc()
            self.potential_kinks = []
    
    # =========================================================================
    # Preview Window
    # =========================================================================

    def schedule_preview_update(self):
        """Schedule a throttled preview window update (~60fps)."""
        if not self.show_preview_window or not self.preview_view:
            return
        
        current_time = time.time()
        time_since_last = current_time - self._last_update_time
        
        if time_since_last >= InterpolateConfig.PREVIEW_UPDATE_INTERVAL:
            self._last_update_time = current_time
            self.update_preview_window()
    
    # =========================================================================
    # Path Interpolation for Preview/Bubble (using GlyphInterpolator + Cache)
    # =========================================================================
    
    def get_interpolated_path_for_glyph(self, glyph):
        """Compute an interpolated bezier path for the given glyph using current master_scalars.
        
        Uses smart caching based on glyph content hash + scalars to avoid redundant computation
        while still reflecting edits immediately.
        """
        if not self.master_scalars:
            return None, 0
        
        try:
            font = self._font if self._font else Glyphs.font
            if not font:
                return None, 0
            
            glyph_id = glyph.name
            scalars_hash = self._cache.get_scalars_hash(self.master_scalars)
            content_hash = self._cache.get_glyph_content_hash(glyph, font)
            
            # Check cache
            cached = self._cache.get_cached_path(glyph_id, scalars_hash, content_hash)
            if cached is not None:
                return cached
            
            # Extract master data using GlyphInterpolator
            glyph_points, widths, reference_layer = GlyphInterpolator.extract_master_data(glyph, font)
            if glyph_points is None:
                return None, 0
            
            if len(widths) != len(self.master_scalars):
                return None, 0
            
            # Validate point compatibility
            if not GlyphInterpolator.validate_points_compatibility(glyph_points):
                return None, 0
            
            all_points = list(glyph_points.values())
            
            # Handle glyphs with no paths (like space)
            if len(all_points[0]) == 0:
                interp_width = GlyphInterpolator.interpolate_width(widths, self.master_scalars)
                result = (None, interp_width)
                self._cache.set_path(glyph_id, scalars_hash, content_hash, result)
                return result
            
            # Interpolate points
            interpolated_points = GlyphInterpolator.interpolate_points(
                glyph_points, self.model, self.master_scalars
            )
            if interpolated_points is None:
                return None, 0
            
            # Interpolate width
            interp_width = GlyphInterpolator.interpolate_width(widths, self.master_scalars)
            
            # Build bezier path (simple version without node tracking)
            displaypath = GlyphInterpolator.build_bezier_path_simple(reference_layer, interpolated_points)
            
            result = (displaypath, interp_width)
            self._cache.set_path(glyph_id, scalars_hash, content_hash, result)
            return result
        except Exception:
            return None, 0
    
    def get_interpolated_kerning(self, leftGlyph, rightGlyph):
        """Get interpolated kerning value between two glyphs using current master_scalars."""
        font = self._font if self._font else Glyphs.font
        return GlyphInterpolator.get_interpolated_kerning(
            leftGlyph, rightGlyph, font, self.master_scalars
        )
    
    def update_preview_window(self):
        """Update the preview window with glyphs from the current line only"""
        if not self.show_preview_window or not self.preview_view:
            return
        
        try:
            font = Glyphs.font
            if not font or not font.currentTab:
                return
            
            # Get layers from the current tab
            allLayers = list(font.currentTab.layers)
            if not allLayers:
                return
            
            # Get the selected layer to know which glyph is selected
            selectedLayers = font.selectedLayers
            if not selectedLayers:
                return
            selectedGlyph = selectedLayers[0].parent
            
            # Find the index of the selected layer in the tab using layersCursor
            selectedIndex = None
            try:
                selectedIndex = font.currentTab.layersCursor
            except Exception:
                # Fallback: find the layer matching the selected glyph
                for i, layer in enumerate(allLayers):
                    if layer and not is_newline_layer(layer) and layer.parent == selectedGlyph:
                        selectedIndex = i
                        break
            
            if selectedIndex is None or selectedIndex >= len(allLayers):
                return
            
            # Find line breaks and determine which line the selected glyph is on
            lineStart = 0
            lineEnd = len(allLayers)
            
            # Find the start of the current line (search backwards for newline)
            for i in range(selectedIndex - 1, -1, -1):
                if is_newline_layer(allLayers[i]):
                    lineStart = i + 1
                    break
            
            # Find the end of the current line (search forwards for newline)
            for i in range(selectedIndex + 1, len(allLayers)):
                if is_newline_layer(allLayers[i]):
                    lineEnd = i
                    break
            
            # Get only the layers for the current line
            lineLayers = allLayers[lineStart:lineEnd]
            
            # Build paths for glyphs in the current line (skip control layers)
            glyphData = []
            prevGlyph = None
            for layer in lineLayers:
                # Skip control layers (newlines, etc.)
                if is_newline_layer(layer):
                    continue
                    
                if layer and layer.parent:
                    glyph = layer.parent
                    path, width = self.get_interpolated_path_for_glyph(glyph)
                    isSelected = (glyph == selectedGlyph)
                    
                    # Calculate interpolated kerning from previous glyph
                    leftKerning = 0
                    if prevGlyph is not None:
                        leftKerning = self.get_interpolated_kerning(prevGlyph, glyph)
                    
                    glyphData.append((path, width, leftKerning, isSelected))
                    prevGlyph = glyph
            
            # Update the view
            self.preview_view.updateWithGlyphs_(glyphData)
        except Exception:
            pass  # Silently handle preview update errors


class AxisSlider(Group):
    def __init__(self, axis, min_value, max_value, posSize=None, owner=None, *args, **kwargs):
        # If posSize is provided, use it; otherwise use "auto"
        if posSize is None:
            posSize = "auto"
        super(AxisSlider, self).__init__(posSize)
        self.axis = axis
        self.owner = owner
        self.callback = kwargs["callback"]

        # Top row: label + slider + value box
        self.label = TextBox(
            (5, 0, 38, 14),
            axis.axisTag,
            sizeStyle="mini",
        )
        self.slider = Slider(
            (45, 0, -50, 14),
            callback=self.update_pos_from_slider,
            minValue=min_value,
            maxValue=max_value,
            value=min_value,
            continuous=True,
            sizeStyle="mini",
        )
        self.valuebox = EditText(
            (-45, 0, 40, 14),
            text=str(min_value),
            sizeStyle="mini",
            callback=self.update_pos_from_text,
        )

        # Second row: play/pause and speed combo
        self.playButton = Button(
            (5, 16, 42, 16),
            "Play",
            sizeStyle="mini",
            callback=self.toggle_play,
        )
        self.speedBox = ComboBox(
            (52, 16, 40, 16),
            ["0.5x", "1x", "2x"],
            sizeStyle="mini",
            callback=self.speed_changed,
            completes=False,
        )
        try:
            self.speedBox.getNSComboBox().setEditable_(False)
            self.speedBox.getNSComboBox().selectItemAtIndex_(1)
            self.speedBox.getNSComboBox().setStringValue_("1x")
        except Exception:
            pass
        try:
            self.speedBox.set("1x")  # Default 1x
        except Exception:
            pass

    def toggle_play(self, sender):
        # Safety check - owner may be invalid during shutdown
        if not self.owner:
            return
        
        is_double = False
        try:
            evt = NSApplication.sharedApplication().currentEvent()
            if evt and hasattr(evt, 'clickCount'):
                is_double = evt.clickCount() >= 2
        except Exception:
            pass

        try:
            if hasattr(self.owner, 'on_axis_play_toggle'):
                self.owner.on_axis_play_toggle(self.axis, is_double)
        except Exception:
            pass

    def speed_changed(self, sender):
        # Safety check - owner may be invalid during shutdown
        if not self.owner:
            return
        
        label = sender.get()
        speed_map = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}
        speed = speed_map.get(label, 1.0)
        try:
            if hasattr(self.owner, 'on_axis_speed_change'):
                self.owner.on_axis_speed_change(self.axis, speed)
        except Exception:
            pass

    def set_play_state(self, is_playing: bool):
        title = "Pause" if is_playing else "Play"
        try:
            self.playButton.getNSButton().setTitle_(title)
        except Exception:
            try:
                self.playButton.setTitle(title)
            except Exception:
                pass

    def set_speed_label(self, speed: float):
        idx = 1
        if speed <= 0.6:
            idx = 0
        elif speed >= 1.5:
            idx = 2
        try:
            self.speedBox.getNSComboBox().selectItemAtIndex_(idx)
            self.speedBox.getNSComboBox().setStringValue_(self.speedBox.getNSComboBox().itemObjectValueAtIndex_(idx))
        except Exception:
            pass

    def get(self):
        return self.slider.get()

    def update_pos_from_text(self, sender):
        # Safety check - owner may be invalid during shutdown
        try:
            value = float(sender.get())
            self.slider.set(value)
            if self.owner and hasattr(self.owner, 'on_axis_scrub'):
                self.owner.on_axis_scrub(self.axis)
            if self.callback:
                self.callback(self.axis, value)
        except (ValueError, TypeError, AttributeError):
            pass  # Ignore invalid text input or invalid owner

    def update_pos_from_slider(self, sender):
        # Safety check - owner may be invalid during shutdown
        try:
            self.valuebox.set(sender.get())
            if self.owner and hasattr(self.owner, 'on_axis_scrub'):
                self.owner.on_axis_scrub(self.axis)
            if self.callback:
                self.callback(self.axis, sender.get())
        except (AttributeError, Exception):
            pass  # Ignore errors during shutdown


# =============================================================================
# Settings Window
# =============================================================================

class InterpolSettings:
    """
    Settings window for Interpol plugin preferences.
    
    Allows users to customize:
    - Tool keyboard shortcut
    - Preview fill color
    - Preview outline/nodes/handles color  
    - Kink indicator color
    - Line style (dotted vs solid)
    - Dekink window preview color
    """
    
    _instance = None  # Singleton instance
    
    def __init__(self):
        self.w = None
        self._build_window()
    
    @classmethod
    def show(cls):
        """Show the settings window (creating it if needed)."""
        if cls._instance is None:
            cls._instance = cls()
        if cls._instance.w is not None:
            cls._instance.w.show()
            cls._instance.w.getNSWindow().makeKeyAndOrderFront_(None)
    
    def _build_window(self):
        """Build the settings window UI."""
        # Window dimensions
        width = 340
        height = 460
        margin = 15
        row_height = 30
        label_width = 180
        color_well_size = 44
        
        self.w = FloatingWindow(
            (width, height),
            "Interpol Settings",
            closable=True,
            autosaveName=KEY + ".settings_window"
        )
        
        y = margin
        
        # --- Tool Shortcut ---
        self.w.shortcutLabel = TextBox(
            (margin, y + 3, label_width, 20),
            "Tool Shortcut Key:"
        )
        current_shortcut = InterpolateConfig.get_tool_shortcut()
        self.w.shortcutField = EditText(
            (margin + label_width, y, 50, 24),
            current_shortcut,
            callback=self._on_shortcut_changed
        )
        self.w.shortcutHint = TextBox(
            (margin + label_width + 55, y + 3, -margin, 20),
            "(restart Glyphs to apply)",
            sizeStyle="small"
        )
        
        y += row_height + 10
        
        # --- Horizontal line separator ---
        self.w.line1 = HorizontalLine((margin, y, -margin, 1))
        y += 15
        
        # --- Interpolation Preview Section ---
        self.w.previewColorsTitle = TextBox(
            (margin, y, -margin, 20),
            "Interpolation Preview",
            sizeStyle="small"
        )
        y += 25
        
        # Fill color
        self.w.fillColorLabel = TextBox(
            (margin, y + 12, label_width, 20),
            "Fill Color:"
        )
        fill_color = InterpolateConfig.get_fill_color()
        self.w.fillColorWell = ColorWell(
            (margin + label_width, y, color_well_size, color_well_size),
            callback=self._on_fill_color_changed,
            color=NSColor.colorWithRed_green_blue_alpha_(*fill_color)
        )
        
        y += color_well_size + 10
        
        # Outline color (for outline, nodes, handles)
        self.w.outlineColorLabel = TextBox(
            (margin, y + 12, label_width, 20),
            "Outline / Nodes / Handles:"
        )
        outline_color = InterpolateConfig.get_outline_color()
        self.w.outlineColorWell = ColorWell(
            (margin + label_width, y, color_well_size, color_well_size),
            callback=self._on_outline_color_changed,
            color=NSColor.colorWithRed_green_blue_alpha_(*outline_color)
        )
        
        y += color_well_size + 10
        
        # Kink indicator color
        self.w.kinkIndicatorColorLabel = TextBox(
            (margin, y + 12, label_width, 20),
            "Kink Indicator:"
        )
        kink_indicator_color = InterpolateConfig.get_kink_indicator_color()
        self.w.kinkIndicatorColorWell = ColorWell(
            (margin + label_width, y, color_well_size, color_well_size),
            callback=self._on_kink_indicator_color_changed,
            color=NSColor.colorWithRed_green_blue_alpha_(*kink_indicator_color)
        )
        
        y += color_well_size + 15
        
        # --- Line Style ---
        self.w.lineStyleLabel = TextBox(
            (margin, y + 2, label_width, 20),
            "Line Style:"
        )
        use_dotted = InterpolateConfig.get_use_dotted_lines()
        self.w.lineStyleRadio = RadioGroup(
            (margin + label_width, y, 120, 40),
            ["Dotted", "Solid"],
            callback=self._on_line_style_changed
        )
        self.w.lineStyleRadio.set(0 if use_dotted else 1)
        
        y += 50
        
        # --- Horizontal line separator ---
        self.w.line2 = HorizontalLine((margin, y, -margin, 1))
        y += 15
        
        # --- Dekink Window Preview Section ---
        self.w.dekinkTitle = TextBox(
            (margin, y, -margin, 20),
            "Dekink Window Preview",
            sizeStyle="small"
        )
        y += 25
        
        # Dekink preview color (for outline, nodes, handles in dekink window)
        self.w.dekinkPreviewColorLabel = TextBox(
            (margin, y + 12, label_width, 20),
            "Outline / Nodes / Handles:"
        )
        dekink_preview_color = InterpolateConfig.get_dekink_preview_color()
        self.w.dekinkPreviewColorWell = ColorWell(
            (margin + label_width, y, color_well_size, color_well_size),
            callback=self._on_dekink_preview_color_changed,
            color=NSColor.colorWithRed_green_blue_alpha_(*dekink_preview_color)
        )
        
        y += color_well_size + 20
        
        # --- Reset to Defaults button ---
        self.w.resetButton = Button(
            (margin, y, -margin, 24),
            "Reset to Defaults",
            callback=self._on_reset_defaults
        )
    
    def _nscolor_to_tuple(self, nscolor) -> Tuple[float, ...]:
        """Convert NSColor to RGBA tuple."""
        try:
            # Try to get RGBA components directly
            color = nscolor.colorUsingColorSpaceName_("NSCalibratedRGBColorSpace")
            if color:
                return (
                    color.redComponent(),
                    color.greenComponent(),
                    color.blueComponent(),
                    color.alphaComponent()
                )
        except Exception:
            pass
        # Fallback
        return (0.0, 0.0, 0.0, 1.0)
    
    def _on_shortcut_changed(self, sender):
        """Handle shortcut field change."""
        value = sender.get().strip().lower()
        if len(value) == 1 and value.isalpha():
            defaults = NSUserDefaults.standardUserDefaults()
            defaults.setObject_forKey_(value, KEY + ".tool_shortcut")
            try:
                defaults.synchronize()
            except Exception:
                pass
            # Note: Tool shortcut change requires Glyphs restart to take effect
    
    def _on_fill_color_changed(self, sender):
        """Handle fill color change."""
        color = self._nscolor_to_tuple(sender.get())
        _save_color_to_defaults(KEY + ".color.fill", color)
        Glyphs.redraw()
    
    def _on_outline_color_changed(self, sender):
        """Handle outline color change."""
        color = self._nscolor_to_tuple(sender.get())
        _save_color_to_defaults(KEY + ".color.outline", color)
        Glyphs.redraw()
    
    def _on_kink_indicator_color_changed(self, sender):
        """Handle kink indicator color change."""
        color = self._nscolor_to_tuple(sender.get())
        _save_color_to_defaults(KEY + ".color.kink_indicator", color)
        Glyphs.redraw()
    
    def _on_line_style_changed(self, sender):
        """Handle line style radio change."""
        use_dotted = sender.get() == 0
        defaults = NSUserDefaults.standardUserDefaults()
        defaults.setBool_forKey_(use_dotted, KEY + ".use_dotted_lines")
        try:
            defaults.synchronize()
        except Exception:
            pass
        Glyphs.redraw()
    
    def _on_dekink_preview_color_changed(self, sender):
        """Handle dekink window preview color change."""
        color = self._nscolor_to_tuple(sender.get())
        _save_color_to_defaults(KEY + ".color.dekink_preview", color)
        Glyphs.redraw()
    
    def _on_reset_defaults(self, sender):
        """Reset all settings to defaults."""
        # Reset shortcut
        self.w.shortcutField.set(DEFAULT_TOOL_SHORTCUT)
        defaults = NSUserDefaults.standardUserDefaults()
        defaults.setObject_forKey_(DEFAULT_TOOL_SHORTCUT, KEY + ".tool_shortcut")
        
        # Reset interpolation preview colors
        self.w.fillColorWell.set(NSColor.colorWithRed_green_blue_alpha_(*DEFAULT_FILL_COLOR))
        _save_color_to_defaults(KEY + ".color.fill", DEFAULT_FILL_COLOR)
        
        self.w.outlineColorWell.set(NSColor.colorWithRed_green_blue_alpha_(*DEFAULT_OUTLINE_COLOR))
        _save_color_to_defaults(KEY + ".color.outline", DEFAULT_OUTLINE_COLOR)
        
        self.w.kinkIndicatorColorWell.set(NSColor.colorWithRed_green_blue_alpha_(*DEFAULT_KINK_INDICATOR_COLOR))
        _save_color_to_defaults(KEY + ".color.kink_indicator", DEFAULT_KINK_INDICATOR_COLOR)
        
        # Reset line style
        self.w.lineStyleRadio.set(0 if DEFAULT_USE_DOTTED_LINES else 1)
        defaults.setBool_forKey_(DEFAULT_USE_DOTTED_LINES, KEY + ".use_dotted_lines")
        
        # Reset dekink window preview color
        self.w.dekinkPreviewColorWell.set(NSColor.colorWithRed_green_blue_alpha_(*DEFAULT_DEKINK_PREVIEW_COLOR))
        _save_color_to_defaults(KEY + ".color.dekink_preview", DEFAULT_DEKINK_PREVIEW_COLOR)
        
        try:
            defaults.synchronize()
        except Exception:
            pass
        
        Glyphs.redraw()


# Reporter Plugin - handles the View menu toggle and drawing
class InterpolateReporter(ReporterPlugin):
    
    def __init__(self):
        # Don't create state in Reporter - it will look up the state from the registry
        self.state = None
        self._last_controller = None  # Track the controller to detect activation
        super(InterpolateReporter, self).__init__()
    
    @objc.python_method
    def _get_state_for_current_font(self):
        """Look up the state instance for the current font from the registry."""
        font = Glyphs.font
        if font is None:
            debug_log("Reporter._get_state_for_current_font(): No font")
            return None
        font_sig = InterpolateState._get_font_signature_static(font)
        state = InterpolateState.get_state_for_font(font)
        if state is None:
            debug_log(f"Reporter._get_state_for_current_font(): No state found for '{font_sig}'")
            debug_log(f"  -> Registry has: {list(InterpolateState._instances.keys())}")
        return state
    
    @objc.python_method
    def settings(self):
        self.menuName = "Interpol Preview"
        debug_log("Reporter settings() called")
    
    @objc.python_method
    def start(self):
        """Called when the plugin is loaded. Adds menu item to Window menu."""
        debug_log("Reporter start() called")
        try:
            # Add "Interpol Settings..." to Window menu
            if Glyphs.buildNumber >= 3320:
                from GlyphsApp.UI import MenuItem
                newMenuItem = MenuItem("Interpol Settings…", action=self.showSettings_, target=self)
            else:
                newMenuItem = NSMenuItem.new()
                newMenuItem.setTitle_("Interpol Settings…")
                newMenuItem.setAction_(self.showSettings_)
                newMenuItem.setTarget_(self)
            Glyphs.menu[WINDOW_MENU].append(newMenuItem)
        except Exception as e:
            debug_log(f"Reporter start() error adding menu item: {e}")
    
    def showSettings_(self, sender):
        """Open the Interpol Settings window."""
        InterpolSettings.show()
    
    @objc.typedSelector(b'v@:@')
    def setController_(self, controller):
        """
        Called when the reporter receives a controller (activated from View menu).
        Font changes are handled by check_font_changed() in update().
        We don't need to clear caches here - just track the controller.
        """
        try:
            # Call parent implementation
            objc.super(InterpolateReporter, self).setController_(controller)
            
            # Just track the controller, don't clear caches
            # Font changes are properly handled by check_font_changed() in update()
            self._last_controller = controller
        except Exception:
            import traceback
            traceback.print_exc()

    @objc.python_method
    def conditionalContextMenus(self):
        """Dynamic context menu items reflecting current toggle state."""
        contextMenus = []
        state = self._get_state_for_current_font()
        if state is None:
            return contextMenus

        def menu_item(name, action, checked=False):
            return {
                'name': name,
                'action': action,
                'state': checked
            }
        label = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Interpol Preview Settings", None, "")
        label.setEnabled_(False)              # non-clickable
        label.setAttributedTitle_(NSAttributedString.alloc().initWithString_attributes_(
            "Interpol Preview Settings", {NSFontAttributeName: NSFont.systemFontOfSize_(12)}
        ))

        # Toggles
        contextMenus.append({'menu': NSMenuItem.separatorItem()})
        contextMenus.append({'menu': label})
        contextMenus.append(menu_item("Show Fill", self.toggleShowFill_, state.show_fill))
        contextMenus.append(menu_item("Show Outlines/Points", self.toggleShowNodes_, state.show_nodes))
        contextMenus.append(menu_item("Show Anchors", self.toggleShowAnchors_, state.show_anchors))
        contextMenus.append(menu_item("Show Kinks", self.toggleShowKinks_, state.show_kinks))
        contextMenus.append({'menu': NSMenuItem.separatorItem()})
        contextMenus.append(menu_item("Dim Layer", self.toggleDimLayer_, state.dim_layer))
        contextMenus.append(menu_item("Follow masters", self.toggleFollowMasters_, state.follow_masters))
        contextMenus.append(menu_item("Spacebar shows interpolation", self.toggleSpacePreviewInterpolated_, state.space_preview_interpolated))
        contextMenus.append({'menu': NSMenuItem.separatorItem()})
        contextMenus.append(menu_item("Preview Window", self.togglePreviewWindow_, state.show_preview_window))
        contextMenus.append(menu_item("Preview Bubble", self.togglePreviewBubble_, state.show_compare_line))

        # Bubble size slider (percentage 50-200) with ticks and live value
        bubble_scale = getattr(state, 'bubble_scale', 1.0) or 1.0
        try:
            container = NSView.alloc().initWithFrame_(((0, 0), (350, 24)))

            slider = NSSlider.alloc().initWithFrame_(((10, 4), (250, 20)))
            slider.cell().setControlSize_(NSControlSizeMini)
            slider.setMinValue_(25)
            slider.setMaxValue_(200)
            slider.setContinuous_(True)
            slider.setNumberOfTickMarks_(8)
            slider.setAllowsTickMarkValuesOnly_(False)
            slider.setTickMarkPosition_(0)  # 0 = below for horizontal sliders
            slider.setDoubleValue_(bubble_scale * 100.0)
            slider.setTarget_(self)
            slider.setAction_(objc.selector(self.bubbleSizeSliderChanged_, signature=b"v@:@"))

            value_field = NSTextField.alloc().initWithFrame_(((265, 4), (50, 20)))
            value_field.setBezeled_(False)
            value_field.setDrawsBackground_(False)
            value_field.setEditable_(False)
            value_field.setSelectable_(False)
            value_field.setAlignment_(2)  # center
            value_field.setIdentifier_("bubbleSizeValue")
            value_field.setStringValue_(f"{int(bubble_scale * 100)}%")

            container.addSubview_(slider)
            container.addSubview_(value_field)

            contextMenus.append(menu_item("Bubble Size", None, False))
            contextMenus.append({'view': container})
        except Exception:
            pass
        contextMenus.append({'menu': NSMenuItem.separatorItem()})

        # Dekink menu item - only show if smooth on-curve points are selected
        try:
            font = Glyphs.font
            if font and font.selectedLayers:
                layer = font.selectedLayers[0]
                selected_smooth_nodes = SynchronizationHelper.get_selected_smooth_oncurve_nodes(layer)
                if selected_smooth_nodes:
                    contextMenus.append(menu_item("Dekink window...", self.openSyncRatiosPanel_, False))
                    contextMenus.append({'menu': NSMenuItem.separatorItem()})
        except Exception:
            pass

        return contextMenus

    def _ensure_palette(self):
        state = self._get_state_for_current_font()
        if state and hasattr(state, '_palette'):
            return state._palette
        return None

    def toggleShowFill_(self, sender=None):
        state = self._get_state_for_current_font()
        if not state:
            return
        # If turning off fill and nodes is already off, turn nodes on instead
        if state.show_fill and not state.show_nodes:
            state.show_fill = False
            state.show_nodes = True
        else:
            state.show_fill = not state.show_fill
        state.save_defaults()
        Glyphs.redraw()

    def toggleShowNodes_(self, sender=None):
        state = self._get_state_for_current_font()
        if not state:
            return
        # If turning off nodes and fill is already off, turn fill on instead
        if state.show_nodes and not state.show_fill:
            state.show_nodes = False
            state.show_fill = True
        else:
            state.show_nodes = not state.show_nodes
        state.save_defaults()
        Glyphs.redraw()

    def toggleShowAnchors_(self, sender=None):
        state = self._get_state_for_current_font()
        if not state:
            return
        state.show_anchors = not state.show_anchors
        state.save_defaults()
        Glyphs.redraw()

    def toggleShowKinks_(self, sender=None):
        state = self._get_state_for_current_font()
        if not state:
            return
        state.show_kinks = not state.show_kinks
        state.save_defaults()
        Glyphs.redraw()

    def togglePreviewWindow_(self, sender=None):
        state = self._get_state_for_current_font()
        if not state:
            return
        palette = self._ensure_palette()
        state.show_preview_window = not state.show_preview_window
        state.save_defaults()
        if palette:
            if state.show_preview_window:
                palette.openPreviewWindow()
            else:
                palette.closePreviewWindow()

    def togglePreviewBubble_(self, sender=None):
        state = self._get_state_for_current_font()
        if not state:
            return
        state.show_compare_line = not state.show_compare_line
        state.save_defaults()
        if state.show_compare_line:
            font = Glyphs.font
            if font:
                state.get_font_bounds(font)
    
    def toggleFollowMasters_(self, sender=None):
        state = self._get_state_for_current_font()
        if not state:
            return
        state.follow_masters = not state.follow_masters
        state.save_defaults()
        Glyphs.redraw()

    def toggleDimLayer_(self, sender=None):
        state = self._get_state_for_current_font()
        if not state:
            return
        state.dim_layer = not state.dim_layer
        state.save_defaults()
        Glyphs.redraw()

    def toggleSpacePreviewInterpolated_(self, sender=None):
        state = self._get_state_for_current_font()
        if not state:
            return
        state.space_preview_interpolated = not state.space_preview_interpolated
        state.save_defaults()
        Glyphs.redraw()

    def bubbleSizeSliderChanged_(self, sender=None):
        state = self._get_state_for_current_font()
        if not state or sender is None:
            return
        raw = sender.doubleValue()
        # Snap to nearest tick if within tolerance (keeps continuous feel)
        tick_values = [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0]
        snap_tol = 5.0  # percentage points
        nearest = min(tick_values, key=lambda v: abs(v - raw))
        if abs(nearest - raw) <= snap_tol:
            raw = nearest
            try:
                sender.setDoubleValue_(raw)
            except Exception:
                pass
        scale_val = max(0.1, raw / 100.0)
        state.bubble_scale = scale_val
        state.save_defaults()

        # Update label next to the slider if present
        try:
            superview = sender.superview()
            if superview:
                for sub in superview.subviews():
                    try:
                        if hasattr(sub, 'identifier') and sub.identifier() == "bubbleSizeValue":
                            sub.setStringValue_(f"{int(raw)}%")
                            break
                    except Exception:
                        pass
        except Exception:
            pass

        Glyphs.redraw()

    def openSyncRatiosPanel_(self, sender=None):
        """Open the Dekink panel for synchronizing point ratios across masters."""
        try:
            SyncRatiosPanel.show_panel()
        except Exception as e:
            if DEBUG_INTERPOLATE:
                print(f"openSyncRatiosPanel_ error: {e}")
                import traceback
                traceback.print_exc()

    
    @objc.python_method
    def background(self, layer):
        # Mark the Reporter as active (Palette checks this to skip work when inactive)
        InterpolateState._reporter_active = True
        InterpolateState._reporter_last_active_time = time.time()
    
    @objc.python_method
    def foreground(self, layer):
        """Draw dim layer overlay and interpolation preview.
        
        Drawing order:
        1. Semi-transparent overlay to dim the main layer (if enabled)
        2. Interpolation preview on top (unaffected by dimming)
        
        When spacebar is pressed with space_preview_interpolated enabled:
        - The main outlines are hidden via needsExtraMainOutlineDrawing methods
        - Interpolated glyphs are drawn in foregroundInViewCoords instead
        """
        # Skip normal drawing when spacebar preview is active
        if self._is_space_pressed():
            return
        
        state = self._get_state_for_current_font()
        if state is None:
            return
        
        # --- Part 1: Draw dim layer overlay ---
        if state.dim_layer:
            self._draw_dim_overlay(layer)
        
        # --- Part 2: Draw interpolation preview on top ---
        self._draw_interpolation_preview(layer, state)
    
    @objc.python_method
    def _is_space_pressed(self):
        """Check if spacebar is currently pressed (for preview mode)."""
        try:
            from Quartz import CGEventSourceKeyState, kCGEventSourceStateHIDSystemState
            # Key code 49 is the spacebar
            return CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, 49)
        except:
            return False
    
    @objc.python_method
    def _should_draw_interpolated_preview(self):
        """Check if we should draw interpolated preview (spacebar + option enabled)."""
        if not self._is_space_pressed():
            return False
        state = self._get_state_for_current_font()
        return state and state.space_preview_interpolated
    
    def needsExtraMainOutlineDrawingForInactiveLayer_(self, layer):
        """Return True - we don't hide inactive layers, we cover them."""
        # Always draw inactive layers normally - we'll cover current line with white + interpolated
        return True
    
    def needsExtraMainOutlineDrawingForActiveLayer_(self, layer):
        """Return False to hide active glyph outline when showing interpolated preview."""
        if self._should_draw_interpolated_preview():
            return False
        return True
    
    @objc.python_method
    def inactiveLayerForeground(self, layer):
        """Not used for spacebar preview - handled in foregroundInViewCoords."""
        pass

    @objc.python_method
    def _draw_dim_overlay(self, layer):
        """Draw semi-transparent shapes over the layer to dim it visually.
        
        Uses a transparency layer to avoid additive transparency where shapes overlap.
        Saves/restores graphics state so alpha doesn't affect subsequent drawing.
        """
        # Use Quartz transparency layer to group all drawing
        # This prevents overlapping shapes from accumulating transparency
        try:
            from Quartz import (
                CGContextBeginTransparencyLayer, CGContextEndTransparencyLayer, 
                CGContextSetAlpha, CGContextSaveGState, CGContextRestoreGState
            )
            context = NSGraphicsContext.currentContext().CGContext()
            
            # Save graphics state so alpha doesn't leak to subsequent drawing
            CGContextSaveGState(context)
            
            # Set alpha for the entire group, then begin transparency layer
            CGContextSetAlpha(context, 0.65)
            CGContextBeginTransparencyLayer(context, None)
            useTransparencyLayer = True
        except:
            # Fallback if Quartz not available
            useTransparencyLayer = False
            context = None
        
        # Draw with opaque white if using transparency layer (alpha applied to whole group)
        # Otherwise use semi-transparent color directly
        if useTransparencyLayer:
            NSColor.whiteColor().set()
        else:
            NSColor.colorWithWhite_alpha_(1.0, 0.65).set()
        
        # Draw over the filled paths
        bezierPath = layer.bezierPath
        if bezierPath:
            bezierPath.setLineWidth_(1.5)
            bezierPath.stroke()
            # bezierPath.fill()
        
        # Draw over open paths with stroke
        openPath = layer.openBezierPath
        if openPath:
            openPath.setLineWidth_(1.5)
            openPath.stroke()
        
        # Get scale for size-independent drawing
        scale = 1.0
        try:
            scale = Glyphs.font.currentTab.scale
        except:
            pass
        
        # Node and handle sizes (scaled)
        nodeRadius = 7.0 / scale
        handleRadius = 5.0 / scale
        handleLineWidth = 2.0 / scale
        
        # Draw over all nodes and handles
        for path in layer.paths:
            for node in path.nodes:
                x, y = node.position.x, node.position.y
                
                if node.type == "offcurve":
                    # Draw circle over handle
                    handleRect = NSMakeRect(
                        x - handleRadius, y - handleRadius,
                        handleRadius * 2, handleRadius * 2
                    )
                    NSBezierPath.bezierPathWithOvalInRect_(handleRect).fill()
                else:
                    # Draw over on-curve node (larger)
                    nodeRect = NSMakeRect(
                        x - nodeRadius, y - nodeRadius,
                        nodeRadius * 2, nodeRadius * 2
                    )
                    # Use square for corner nodes, circle for smooth
                    if node.type == "line" or node.connection == "sharp":
                        NSBezierPath.fillRect_(nodeRect)
                    else:
                        NSBezierPath.bezierPathWithOvalInRect_(nodeRect).fill()
                    
                    # Also draw over handle lines connecting to this node
                    prevNode = node.prevNode
                    nextNode = node.nextNode
                    handleLine = NSBezierPath.bezierPath()
                    handleLine.setLineWidth_(handleLineWidth)
                    
                    if prevNode and prevNode.type == "offcurve":
                        handleLine.moveToPoint_((x, y))
                        handleLine.lineToPoint_((prevNode.position.x, prevNode.position.y))
                    if nextNode and nextNode.type == "offcurve":
                        handleLine.moveToPoint_((x, y))
                        handleLine.lineToPoint_((nextNode.position.x, nextNode.position.y))
                    
                    handleLine.stroke()
        
        # End transparency layer and restore graphics state
        if useTransparencyLayer:
            CGContextEndTransparencyLayer(context)
            CGContextRestoreGState(context)
    
    @objc.python_method
    def _draw_interpolation_preview(self, layer, state):
        """Draw the interpolation preview (fill, nodes, handles, anchors, kinks)."""
        # Debug: show what location the Reporter is drawing (only when debug enabled)
        if DEBUG_INTERPOLATE and state.current_location and state.axis_min_max:
            loc_info = [f"{tag}={state.get_actual_value(tag):.1f}" for tag in state.current_location if tag in state.axis_min_max]
            print(f"[Interpol] Reporter.foreground(): Drawing at location: {', '.join(loc_info)}, state id={id(state)}")
        
        # Cache state properties for hot path
        show_kinks = state.show_kinks
        show_preview = state.show_preview
        show_fill = state.show_fill
        show_nodes = state.show_nodes
        show_anchors = state.show_anchors
        
        # Get the scale factor to keep sizes consistent at all zoom levels
        try:
            scale = self.getScale()
        except Exception:
            scale = 1.0
        
        # Calculate offset for centering interpolation within layer
        layerCenterX = layer.width / 2.0
        interpWidth = getattr(state, 'interpolated_width', layer.width)
        interpCenterX = interpWidth / 2.0
        offsetX = layerCenterX - interpCenterX
        
        # Draw kink indicators at interpolated positions with our styled circles
        if show_kinks and getattr(state, "interpolated_kinks", None):
            # Draw kink indicators using our styled DrawingHelpers
            for x, y, current_severity, max_severity, current_angle_deg in state.interpolated_kinks:
                centeredX = x + offsetX
                DrawingHelpers.draw_kink_indicator(
                    (centeredX, y),
                    current_severity,
                    max_severity,
                    current_angle_deg,
                    scale
                )
        
        if not show_preview:
            return
        
        # Check if we have a valid displaypath with elements
        displaypath = getattr(state, "displaypath", None)
        if not displaypath or displaypath.elementCount() == 0:
            return
        
        # Get configurable colors and line style
        fill_color = InterpolateConfig.get_fill_color()
        outline_color = InterpolateConfig.get_outline_color()
        use_dotted = InterpolateConfig.get_use_dotted_lines()
        
        if show_fill:
            fillPath = displaypath.copy()
            fillTransform = NSAffineTransform.transform()
            fillTransform.translateXBy_yBy_(offsetX, 0)
            fillPath.transformUsingAffineTransform_(fillTransform)
            NSColor.colorWithRed_green_blue_alpha_(*fill_color).set()
            fillPath.fill()
        
        if show_nodes:
            strokePath = displaypath.copy()
            strokeTransform = NSAffineTransform.transform()
            strokeTransform.translateXBy_yBy_(offsetX, 0)
            strokePath.transformUsingAffineTransform_(strokeTransform)
            # Use outline color with high alpha for stroke
            stroke_color = (outline_color[0], outline_color[1], outline_color[2], 0.9)
            NSColor.colorWithRed_green_blue_alpha_(*stroke_color).set()
            strokePath.setLineWidth_(1.4 / scale)
            if use_dotted:
                strokePath.setLineDash_count_phase_([1.5 / scale, 2.0 / scale], 2, 0.0)
            strokePath.stroke()
        
        if show_nodes and getattr(state, "interp_nodes", None):
            # Draw nodes with white fill and colored outline
            nodeColor = outline_color
            
            for on_curve, pt in state.interp_nodes:
                # Apply centering offset
                centeredX = pt.x + offsetX
                
                # Draw outer circle with colored outline
                DrawingHelpers.draw_circle(
                    (centeredX, pt.y), 4.0,
                    fill_color=(1.0, 1.0, 1.0, 0.75),
                    stroke_color=nodeColor,
                    stroke_width=1.0,
                    scale=scale
                )
                
                # Draw center dot
                DrawingHelpers.draw_circle(
                    (centeredX, pt.y), 1.0,
                    fill_color=nodeColor,
                    scale=scale
                )
        
        if show_nodes and getattr(state, "interp_handles", None):
            # Draw handle connecting lines
            handleColor = outline_color
            for handle, oncurve in state.interp_handles:
                DrawingHelpers.draw_dashed_line(
                    (handle.x + offsetX, handle.y),
                    (oncurve.x + offsetX, oncurve.y),
                    handleColor,
                    line_width=0.8,
                    dash_pattern=(1.0, 1.5),
                    scale=scale,
                    use_dash=use_dotted
                )
            
            # Draw handle dots
            for handle, oncurve in state.interp_handles:
                DrawingHelpers.draw_circle(
                    (handle.x + offsetX, handle.y), 2.5,
                    fill_color=handleColor,
                    scale=scale
                )
        
        if show_anchors and getattr(state, "interpolated_anchors", None):
            # Draw anchors as bright red X crosses
            anchorColor = InterpolateConfig.ANCHOR_COLOR
            for name, pt in state.interpolated_anchors.items():
                centeredX = pt[0] + offsetX
                DrawingHelpers.draw_cross(
                    (centeredX, pt[1]), 5.0,
                    anchorColor,
                    stroke_width=1.25,
                    scale=scale
                )
        
        # Draw sync preview if active
        self._draw_sync_preview(layer, state, scale, offsetX)
    
    @objc.python_method
    def _draw_sync_preview(self, layer, state, scale, offsetX):
        """Draw the dekink preview in green to show what synchronization will look like."""
        if not SyncRatiosPanel.is_preview_active():
            return
        
        # Check if we're on the right glyph
        glyph = layer.parent
        if not glyph or glyph.name != SyncRatiosPanel.get_preview_glyph_name():
            return
        
        # Validate preview is still fresh (not stale from undo/redo/edits)
        font = state._font if state._font else Glyphs.font
        if SyncRatiosPanel.invalidate_preview_if_stale(glyph, font):
            # Preview was stale and has been cleared, nothing to draw
            return
        
        preview_points = SyncRatiosPanel.get_preview_points()
        if not preview_points:
            return
        
        # Check we have a model and scalars
        if not state.model or not state.master_scalars:
            return
        
        try:
            if not font:
                return
            
            # Validate preview points have same structure as master_scalars
            if len(preview_points) != len(state.master_scalars):
                # Structure mismatch - preview is stale
                SyncRatiosPanel._clear_preview_state()
                return
            
            # Validate point compatibility
            if not GlyphInterpolator.validate_points_compatibility(preview_points):
                SyncRatiosPanel._clear_preview_state()
                return
            
            # Interpolate the preview points
            interpolated_preview = GlyphInterpolator.interpolate_points(
                preview_points, state.model, state.master_scalars
            )
            if interpolated_preview is None:
                return
            
            # Get reference layer for building the bezier path
            first_master_layer = glyph.layers[font.masters[0].id]
            reference_decomposed = GlyphInterpolator.get_decomposed_layer(first_master_layer)
            
            # Build the preview path with node/handle tracking (same as main interpolation preview)
            preview_path, sync_nodes, sync_handles = GlyphInterpolator.build_bezier_path(
                reference_decomposed, interpolated_preview
            )
            if not preview_path or preview_path.elementCount() == 0:
                return
            
            # Get configurable dekink preview colors
            dekink_colors = InterpolateConfig.get_dekink_preview_colors()
            
            # Draw the preview
            previewTransform = NSAffineTransform.transform()
            previewTransform.translateXBy_yBy_(offsetX, 0)
            
            # Draw filled preview
            if state.show_fill:
                previewFillPath = preview_path.copy()
                previewFillPath.transformUsingAffineTransform_(previewTransform)
                NSColor.colorWithRed_green_blue_alpha_(*dekink_colors['fill']).set()
                previewFillPath.fill()
            
            # Draw stroked preview (outline)
            previewStrokePath = preview_path.copy()
            previewStrokePath.transformUsingAffineTransform_(previewTransform)
            NSColor.colorWithRed_green_blue_alpha_(*dekink_colors['stroke']).set()
            previewStrokePath.setLineWidth_(1.5 / scale)
            previewStrokePath.stroke()
            
            # Draw handles and points using DrawingHelpers (same approach as main interpolation preview)
            self._draw_sync_preview_handles_and_nodes(sync_nodes, sync_handles, offsetX, scale, dekink_colors)
            
        except Exception as e:
            if DEBUG_INTERPOLATE:
                print(f"_draw_sync_preview error: {e}")
                import traceback
                traceback.print_exc()
    
    @objc.python_method
    def _draw_sync_preview_handles_and_nodes(self, sync_nodes, sync_handles, offsetX, scale, colors=None):
        """Draw handles and nodes for the sync preview using DrawingHelpers."""
        try:
            # Use provided colors or fall back to defaults
            if colors:
                nodeColor = colors['node']
                handleColor = colors['handle']
                handleLineColor = colors['handle_line']
            else:
                nodeColor = InterpolateConfig.SYNC_PREVIEW_NODE_COLOR
                handleColor = InterpolateConfig.SYNC_PREVIEW_HANDLE_COLOR
                handleLineColor = InterpolateConfig.SYNC_PREVIEW_HANDLE_LINE_COLOR
            
            # Draw handle connecting lines (same pattern as main interpolation preview)
            for handle, oncurve in sync_handles:
                DrawingHelpers.draw_dashed_line(
                    (handle.x + offsetX, handle.y),
                    (oncurve.x + offsetX, oncurve.y),
                    handleLineColor,
                    line_width=0.8,
                    dash_pattern=(1.0, 1.5),
                    scale=scale
                )
            
            # Draw handle dots
            for handle, oncurve in sync_handles:
                DrawingHelpers.draw_circle(
                    (handle.x + offsetX, handle.y), 2.5,
                    fill_color=handleColor,
                    scale=scale
                )
            
            # Draw on-curve nodes
            for on_curve, pt in sync_nodes:
                # Draw outer circle with colored outline
                DrawingHelpers.draw_circle(
                    (pt.x + offsetX, pt.y), 4.0,
                    fill_color=(1.0, 1.0, 1.0, 0.75),
                    stroke_color=nodeColor,
                    stroke_width=1.0,
                    scale=scale
                )
                
                # Draw center dot
                DrawingHelpers.draw_circle(
                    (pt.x + offsetX, pt.y), 1.0,
                    fill_color=nodeColor,
                    scale=scale
                )
            
        except Exception as e:
            if DEBUG_INTERPOLATE:
                print(f"_draw_sync_preview_handles_and_nodes error: {e}")
                import traceback
                traceback.print_exc()
    
    @objc.python_method
    def _draw_spacebar_interpolated_line(self, state):
        """Draw interpolated glyphs for the current line when spacebar is pressed.
        
        Uses interpolated widths and kerning for accurate spacing.
        Left-aligns the line by calculating line start from ORIGINAL layer positions,
        then draws with interpolated metrics.
        """
        try:
            font = Glyphs.font
            if not font or not font.currentTab:
                return
            
            tab = font.currentTab
            scale = tab.scale
            
            # Get cursor position and line info
            cursorIndex = tab.layersCursor
            allLayers = tab.composedLayers
            if not allLayers or cursorIndex >= len(allLayers):
                return
            
            # Find line boundaries (between newlines)
            lineStart = cursorIndex
            while lineStart > 0:
                if is_newline_layer(allLayers[lineStart - 1]):
                    break
                lineStart -= 1
            
            lineEnd = cursorIndex
            while lineEnd < len(allLayers) - 1:
                if is_newline_layer(allLayers[lineEnd + 1]):
                    break
                lineEnd += 1
            
            # Get layers for current line
            lineLayers = allLayers[lineStart:lineEnd + 1]
            if not lineLayers:
                return
            
            # Get selectedLayerOrigin - this is where the selected glyph starts
            selectedOrigin = tab.selectedLayerOrigin
            
            # Calculate line start X using ORIGINAL layer widths (for left alignment)
            # This finds where Glyphs positions the first glyph
            lineStartX = selectedOrigin.x
            for i in range(cursorIndex - 1, lineStart - 1, -1):
                lyr = allLayers[i]
                if not lyr or is_newline_layer(lyr):
                    continue
                # Use ORIGINAL layer width for positioning
                lineStartX -= lyr.width * scale
                # Use ORIGINAL kerning for positioning
                if i + 1 <= cursorIndex:
                    nextLyr = allLayers[i + 1]
                    if (nextLyr and not is_newline_layer(nextLyr) and 
                        lyr.parent and nextLyr.parent):
                        # Get kerning from current master
                        kern = font.kerningForPair(font.selectedFontMaster.id, 
                                                   lyr.parent.rightKerningKey,
                                                   nextLyr.parent.leftKerningKey)
                        if kern:
                            lineStartX -= kern * scale
            
            # Get font metrics for white coverage height
            descender, ascender, _ = self._get_font_metrics(font, state)
            
            # Extra padding to cover nodes, handles, and any UI elements
            nodePadding = 15 / scale  # Account for node circles, handles etc.
            
            # First pass: Draw white rectangles to cover original glyphs on current line
            # Use ORIGINAL positions and widths, but expand to cover actual bounds
            NSColor.whiteColor().set()
            currentX = lineStartX
            for i, lyr in enumerate(lineLayers):
                if not lyr or is_newline_layer(lyr):
                    continue
                
                glyph = lyr.parent if lyr.parent else None
                layerWidth = lyr.width * scale
                
                # Apply original kerning
                if i > 0:
                    prevLyr = lineLayers[i - 1]
                    if (prevLyr and not is_newline_layer(prevLyr) and
                        prevLyr.parent and glyph):
                        kern = font.kerningForPair(font.selectedFontMaster.id,
                                                   prevLyr.parent.rightKerningKey,
                                                   glyph.leftKerningKey if glyph else None)
                        if kern:
                            currentX += kern * scale
                
                # Get actual layer bounds to cover overhanging parts
                bounds = lyr.bounds
                if bounds and bounds.size.width > 0:
                    # Calculate coverage rect from actual bounds + padding
                    boundsMinX = bounds.origin.x - nodePadding
                    boundsMaxX = bounds.origin.x + bounds.size.width + nodePadding
                    boundsMinY = bounds.origin.y - nodePadding
                    boundsMaxY = bounds.origin.y + bounds.size.height + nodePadding
                    
                    # Also ensure we cover the full advance width
                    coverMinX = min(boundsMinX, -nodePadding)
                    coverMaxX = max(boundsMaxX, lyr.width + nodePadding)
                    coverMinY = min(boundsMinY, descender - nodePadding)
                    coverMaxY = max(boundsMaxY, ascender + nodePadding)
                    
                    # Draw white rectangle with actual bounds + padding
                    coverageRect = NSMakeRect(
                        currentX + coverMinX * scale,
                        selectedOrigin.y + coverMinY * scale,
                        (coverMaxX - coverMinX) * scale,
                        (coverMaxY - coverMinY) * scale
                    )
                else:
                    # Fallback: use advance width with generous vertical coverage
                    coverageRect = NSMakeRect(
                        currentX - nodePadding * scale,
                        selectedOrigin.y + (descender - nodePadding) * scale,
                        layerWidth + 2 * nodePadding * scale,
                        (ascender - descender + 2 * nodePadding) * scale
                    )
                
                NSBezierPath.fillRect_(coverageRect)
                
                currentX += layerWidth
            
            # Second pass: Draw interpolated glyphs with interpolated widths/kerning
            glyphData = []
            currentX = lineStartX  # Start from same position
            prevGlyph = None
            
            # Track selected glyph center to keep preview stable
            selectedGlyphX = None
            selectedGlyphInterpWidth = None
            selectedLayer = allLayers[cursorIndex] if cursorIndex < len(allLayers) else None
            selectedLayerWidth = selectedLayer.width if selectedLayer else 0
            
            for i, lyr in enumerate(lineLayers):
                if not lyr or is_newline_layer(lyr):
                    continue
                
                glyph = lyr.parent if lyr.parent else None
                layerIndex = lineStart + i  # Absolute index within composed layers
                isSelected = (layerIndex == cursorIndex)
                
                if not glyph:
                    # Space or control character - use layer width
                    if isSelected:
                        selectedGlyphX = currentX
                        selectedGlyphInterpWidth = lyr.width * scale
                    currentX += lyr.width * scale
                    prevGlyph = None
                    continue
                
                # Apply INTERPOLATED kerning from previous glyph
                if prevGlyph is not None:
                    kern = state.get_interpolated_kerning(prevGlyph, glyph)
                    currentX += kern * scale
                
                # Get interpolated path and width
                result = state.get_interpolated_path_for_glyph(glyph)
                if result:
                    path, width = result
                    if path:
                        glyphData.append((path, currentX))
                    # Track selected glyph position and width (scaled) for centering
                    if isSelected:
                        selectedGlyphX = currentX
                        selectedGlyphInterpWidth = width * scale
                    # Advance by INTERPOLATED width
                    currentX += width * scale
                else:
                    if isSelected:
                        selectedGlyphX = currentX
                        selectedGlyphInterpWidth = lyr.width * scale
                    currentX += lyr.width * scale
                
                prevGlyph = glyph
            
            # Compute offset to keep selected glyph centered at its original center
            offsetX = 0
            if selectedGlyphX is not None and selectedLayerWidth:
                originalCenter = selectedOrigin.x + (selectedLayerWidth * scale) * 0.5
                interpCenter = selectedGlyphX + (selectedGlyphInterpWidth or (selectedLayerWidth * scale)) * 0.5
                offsetX = originalCenter - interpCenter
            
            # Draw all interpolated glyphs in solid black
            NSColor.blackColor().set()
            
            for path, xPos in glyphData:
                transform = NSAffineTransform.transform()
                transform.translateXBy_yBy_(xPos + offsetX, selectedOrigin.y)
                transform.scaleBy_(scale)
                
                transformedPath = path.copy()
                transformedPath.transformUsingAffineTransform_(transform)
                transformedPath.fill()
                
        except Exception as e:
            if DEBUG_INTERPOLATE:
                print(f"_draw_spacebar_interpolated_line error: {e}")
                import traceback
                traceback.print_exc()

    @objc.python_method
    def _get_current_line_info(self, tab, cursorIndex):
        """
        Get information about the current line in the edit tab.
        
        Args:
            tab: The current edit tab
            cursorIndex: Position of the cursor in the layers list
            
        Returns:
            Tuple of (lineLayers, lineStart, lineEnd) or None if invalid
        """
        allLayers = list(tab.layers)
        if not allLayers or cursorIndex is None or cursorIndex >= len(allLayers):
            return None
        
        # Find the start of the current line (after previous newline)
        lineStart = 0
        for i in range(cursorIndex - 1, -1, -1):
            if is_newline_layer(allLayers[i]):
                lineStart = i + 1
                break
        
        # Find the end of the current line (before next newline)
        lineEnd = len(allLayers)
        for i in range(cursorIndex + 1, len(allLayers)):
            if is_newline_layer(allLayers[i]):
                lineEnd = i
                break
        
        # Get non-newline layers for the current line
        lineLayers = [l for l in allLayers[lineStart:lineEnd] if not is_newline_layer(l)]
        if not lineLayers:
            return None
        
        return (lineLayers, lineStart, lineEnd, allLayers)
    
    @objc.python_method
    def _get_font_metrics(self, font, state=None):
        """
        Get font metrics with caching for performance.
        
        Args:
            font: GSFont instance
            state: Optional InterpolateState for cache access
            
        Returns:
            Tuple of (descender, ascender, upm)
        """
        # Try cache first if state is available
        if state:
            font_sig = InterpolateState._get_font_signature_static(font)
            cached = state._cache.get_font_metrics(font_sig)
            if cached is not None:
                return cached
        
        # Calculate metrics
        upm = font.upm
        descender = -250
        ascender = 750
        if font.masters:
            master = font.masters[0]
            descender = getattr(master, 'descender', -250)
            ascender = getattr(master, 'ascender', 750)
        
        metrics = (descender, ascender, upm)
        
        # Cache result if state is available
        if state:
            state._cache.set_font_metrics(font_sig, metrics)
        
        return metrics
    
    @objc.python_method
    def foregroundInViewCoords(self):
        """Draw interpolated comparison line as a bubble overlay, or spacebar preview."""
        state = self._get_state_for_current_font()
        if state is None:
            return
        
        # Handle spacebar interpolated preview
        if self._should_draw_interpolated_preview():
            self._draw_spacebar_interpolated_line(state)
            return
        
        # Handle bubble preview
        if not state.show_compare_line:
            return
        
        try:
            font = Glyphs.font
            if not font or not font.currentTab:
                return
            
            tab = font.currentTab
            viewScale = tab.scale  # View zoom level
            
            # Bubble scale combines view zoom with user's size preference
            bubbleScale = state.bubble_scale
            scale = viewScale * bubbleScale
            
            # Get cursor position
            try:
                cursorIndex = tab.layersCursor
            except Exception:
                return
            
            # Get current line info using helper
            lineInfo = self._get_current_line_info(tab, cursorIndex)
            if lineInfo is None:
                return
            lineLayers, lineStart, lineEnd, allLayers = lineInfo
            
            # Get the selected layer info
            selectedLayers = font.selectedLayers
            if not selectedLayers:
                return
            
            # Get font metrics using helper (with caching)
            descender, ascender, upm = self._get_font_metrics(font, state)
            
            # Get selectedLayerOrigin for positioning
            selectedOrigin = tab.selectedLayerOrigin
            
            # Get the currently selected layer to align bubble with it
            selectedLayer = allLayers[cursorIndex] if cursorIndex < len(allLayers) else None

            # Calculate X position where the selected glyph starts in the line
            # This will be our alignment point (use viewScale for positioning, not bubble scale)
            selectedGlyphStartX = selectedOrigin.x
            
            # Calculate the X start position for the full line (for drawing all glyphs)
            # Use scale (which includes bubbleScale) for bubble content positioning
            lineStartX = selectedOrigin.x
            for i in range(cursorIndex - 1, lineStart - 1, -1):
                lyr = allLayers[i]
                if lyr and not is_newline_layer(lyr):
                    lineStartX -= lyr.width * scale
                    # Account for kerning with next glyph (only if both have parents)
                    if i + 1 < len(allLayers) and i + 1 <= cursorIndex:
                        nextLyr = allLayers[i + 1]
                        if (nextLyr and not is_newline_layer(nextLyr) and 
                            lyr.parent and nextLyr.parent):
                            kern = state.get_interpolated_kerning(lyr.parent, nextLyr.parent)
                            lineStartX -= kern * scale
            
            # Calculate the Y position for the bubble (one line below current)
            # Use the font's EditView Line Height if set, otherwise calculate from metrics
            editViewLineHeight = None
            try:
                editViewLineHeight = font.customParameters["EditView Line Height"]
            except (KeyError, TypeError):
                pass
            
            if editViewLineHeight:
                lineHeight = editViewLineHeight * scale
            else:
                # Default line height: ascender - descender + some leading
                lineHeight = (ascender - descender) * 1.2 * scale
            
            # Position bubble one line below current line
            bubbleY = selectedOrigin.y - lineHeight
            
            # Get cached font-wide bounds for performance (only calculate once)
            font_sig = InterpolateState._get_font_signature_static(font)
            cached_bounds = state._cache.get_font_bounds(font_sig)
            if cached_bounds is None:
                fontMinY = descender
                fontMaxY = ascender
            else:
                fontMinY, fontMaxY = cached_bounds
            
            # First pass: collect interpolated paths and calculate actual bounding box
            # Also track where the selected glyph is positioned in the bubble
            glyphData = []  # List of (glyph, path, width, xPos) tuples
            currentX = lineStartX
            prevGlyph = None
            minY = float('inf')
            maxY = float('-inf')
            totalWidth = 0
            selectedGlyphX = None  # X position of selected glyph in bubble
            selectedGlyphWidth = 0  # Width of selected glyph for calculating center
            selectedGlyphLayer = None  # The actual layer of the selected glyph
            
            for i, lyr in enumerate(lineLayers):
                # Handle spaces and other glyphs without outlines
                if not lyr:
                    continue
                
                # Check if this is the selected glyph
                layerIndex = lineStart + i
                isSelectedGlyph = (layerIndex == cursorIndex)
                
                glyph = lyr.parent if lyr.parent else None
                
                # For spaces or glyphs without parents, just add width
                if not glyph:
                    # Track selected glyph position even for spaces
                    if isSelectedGlyph:
                        selectedGlyphX = currentX
                        selectedGlyphWidth = lyr.width * scale
                        selectedGlyphLayer = lyr
                    currentX += lyr.width * scale
                    totalWidth += lyr.width
                    # Don't update prevGlyph for spaces (no kerning tracking)
                    continue
                
                # Track selected glyph position and width
                if isSelectedGlyph:
                    selectedGlyphX = currentX
                    selectedGlyphLayer = lyr
                
                # Apply kerning
                if prevGlyph is not None:
                    kern = state.get_interpolated_kerning(prevGlyph, glyph)
                    currentX += kern * scale
                    totalWidth += kern
                
                # Get interpolated path for this glyph
                path, width = state.get_interpolated_path_for_glyph(glyph)
                
                # Track width of selected glyph
                if isSelectedGlyph:
                    selectedGlyphWidth = width * scale
                
                if path:
                    # Store for drawing later
                    glyphData.append((glyph, path, width, currentX))
                    
                    # Get bounds of interpolated path to calculate actual height
                    bounds = path.bounds()
                    if bounds.size.width > 0 and bounds.size.height > 0:
                        pathMinY = bounds.origin.y
                        pathMaxY = bounds.origin.y + bounds.size.height
                        minY = min(minY, pathMinY)
                        maxY = max(maxY, pathMaxY)
                
                currentX += width * scale
                totalWidth += width
                prevGlyph = glyph
            
            # Calculate horizontal offset to align bubble with selected glyph
            # The selected glyph in the bubble should appear at selectedGlyphStartX
            if selectedGlyphX is not None:
                bubbleOffsetX = selectedGlyphStartX - selectedGlyphX
            else:
                bubbleOffsetX = 0
            
            # Use the more extreme values between interpolated glyphs and font-wide bounds
            # This ensures the bubble is at least as tall as the font's tallest/deepest glyphs
            minY = min(minY if minY != float('inf') else fontMinY, fontMinY)
            maxY = max(maxY if maxY != float('-inf') else fontMaxY, fontMaxY)
            
            # Don't draw bubble if there's no width (empty line or only newlines)
            if totalWidth <= 0:
                return
            
            # Calculate bubble dimensions with generous margins
            verticalMargin = 30 * scale
            horizontalMargin = 100 * scale  # Generous horizontal margins for breathing room
            totalWidthScaled = totalWidth * scale
            
            # Calculate the bubble height (scaled content + margins)
            bubbleHeight = (maxY - minY) * scale + 2 * verticalMargin
            
            # Calculate the top of the bubble if we place it at bubbleY
            # The bubble content's maxY (top of glyphs) would be at bubbleY + maxY * scale
            # With margin, the bubble top would be at bubbleY + maxY * scale + verticalMargin
            bubbleTopIfAtBubbleY = bubbleY + maxY * scale + verticalMargin
            
            # The current line's bottom is at selectedOrigin.y + descender (in view coords)
            # Add a gap to ensure no overlap
            gap = 20 * viewScale  # Fixed gap that doesn't scale with bubble size
            currentLineBottom = selectedOrigin.y + descender * viewScale
            
            # If bubble would overlap, push it down
            if bubbleTopIfAtBubbleY > currentLineBottom - gap:
                # Adjust bubbleY so the top of the bubble is below currentLineBottom - gap
                bubbleY = currentLineBottom - gap - maxY * scale - verticalMargin
            
            bgRect = NSMakeRect(
                lineStartX - horizontalMargin + bubbleOffsetX,
                bubbleY + minY * scale - verticalMargin,
                totalWidthScaled + 2 * horizontalMargin,
                bubbleHeight
            )
            
            # Store bubble rect for hit testing (use state if available)
            if state:
                state._bubble_rect = bgRect
            
            # Corner radius for rounded bubble (scales with view)
            cornerRadius = 20 * viewScale
            
            # Triangle size - scales slower than zoom to stay visible when zoomed out
            triangleScale = max(viewScale ** 0.5, 0.5)
            triangleWidth = 24 * triangleScale
            triangleHeight = 15 * triangleScale
            
            # Calculate center of selected glyph in view coordinates
            selectedGlyphCenterX = selectedGlyphStartX
            if selectedGlyphLayer is not None:
                actualWidth = selectedGlyphLayer.width * viewScale
                selectedGlyphCenterX = selectedGlyphStartX + actualWidth / 2
            
            # Triangle position
            triangleTipX = selectedGlyphCenterX
            triangleTipY = bgRect.origin.y + bgRect.size.height  # Top of bubble
            
            # Build bubble path with pointer using helper
            bubbleWithPointer = DrawingHelpers.build_bubble_path(
                bgRect, cornerRadius,
                (triangleTipX, triangleTipY),
                triangleWidth, triangleHeight
            )
            
            # Draw bubble with shadow using helper
            DrawingHelpers.draw_bubble_with_shadow(
                bubbleWithPointer,
                fill_color=(1, 1, 1, 1),
                stroke_color=InterpolateConfig.BUBBLE_BORDER_COLOR,
                stroke_width=1.0
            )
            
            # Draw interpolated glyphs in black
            NSColor.blackColor().set()
            
            for glyph, path, width, xPos in glyphData:
                # Transform and draw the path (apply bubble offset)
                transform = NSAffineTransform.transform()
                transform.translateXBy_yBy_(xPos + bubbleOffsetX, bubbleY)
                transform.scaleBy_(scale)
                
                transformedPath = path.copy()
                transformedPath.transformUsingAffineTransform_(transform)
                transformedPath.fill()
                
        except Exception:
            pass  # Silently handle compare line drawing errors
    
    @objc.python_method
    def __file__(self):
        """Please leave this method unchanged"""
        return __file__


# Palette Plugin - handles the axis sliders UI
class InterpolatePalette(PalettePlugin):
    dialog = objc.IBOutlet()

    def __init__(self):
        # Create a NEW state instance for this palette (not a singleton)
        # Each font window gets its own palette instance with its own state
        self.state = InterpolateState()
        super(InterpolatePalette, self).__init__()

    @objc.python_method
    def settings(self):
        # Create a new state instance for this palette if not already created
        if not hasattr(self, 'state') or self.state is None:
            self.state = InterpolateState()
        self.name = "Interpol"
        width = 160
        # Height will be dynamically adjusted based on number of axes
        # Start with enough room for checkboxes (7 rows × 22px = 154px minimum)
        initial_height = 160
        self.paletteView = Window((width, initial_height))
        debug_log(f"Palette settings() called - instance {id(self)}")
        
        # NOTE: _windowController is NOT available yet in settings()!
        # We must NOT use Glyphs.font as fallback because it might be wrong.
        # Instead, set up a minimal UI and let update() handle proper initialization
        # when _windowController becomes available.
        
        # Initialize state with empty data - will be populated in first update()
        self.state.current_location = {}
        self.state.current_glyph = None
        self.state.glyph_points = {}
        
        # Track the font signature for this specific palette instance
        # Will be set properly in first update() call
        self._instance_font_sig = None
        self._initialized_for_font = False  # Flag to track if we've done proper init
        
        # Build minimal UI structure (axis slots will be updated in first update())
        self.setup_axes()

        # Animation timer reference
        self._animation_timer = None
        
        # Set up height constraint for dynamic resizing
        # Initial height for 7 controls (checkboxes + slider row) with no axes
        self._current_height = 160
        
        # Create height constraint using NSLayoutConstraint
        self.heightConstraint = NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
            self.paletteView.group.getNSView(),
            NSLayoutAttributeHeight,
            NSLayoutRelationEqual,
            None,
            NSLayoutAttributeNotAnAttribute,
            1,
            self._current_height,
        )
        
        # Required: set self.dialog to the NSView
        self.dialog = self.paletteView.group.getNSView()
        self.dialog.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.dialog.addConstraint_(self.heightConstraint)
    
    def minHeight(self):
        """Return minimum height for the palette - Glyphs uses this for sizing."""
        return getattr(self, '_current_height', 160)
    
    def maxHeight(self):
        """Return maximum height for the palette - allows user resizing up to this."""
        # Allow resizing up to 500px for fonts with many axes
        return 500

    @objc.python_method
    def populate_axis_data(self, font=None):
        """Populate axis_min_max data without rebuilding UI. Safe to call from update()."""
        # Use provided font, or get from windowController, or fall back to Glyphs.font
        if font is None:
            if hasattr(self, '_windowController') and self._windowController:
                font = self._windowController.documentFont()
            else:
                font = Glyphs.font
        
        if not font or not font.axes:
            debug_log(f"populate_axis_data(): No font or no axes (font={font is not None}, axes={len(font.axes) if font and font.axes else 0})")
            return
        
        font_name = getattr(font, 'familyName', 'Unknown')
        axes = font.axes
        masters = font.masters
        debug_log(f"populate_axis_data(): Populating for font '{font_name}' with {len(axes)} axes")
        
        self.state.axis_min_max = {}
        axis_tags = []  # Collect axis tags for default tool assignment
        for i, axis in enumerate(axes):
            min_value = min(
                master.internalAxesValues[i] for master in masters
            )
            max_value = max(
                master.internalAxesValues[i] for master in masters
            )
            self.state.axis_min_max[axis.axisTag] = (min_value, max_value)
            axis_tags.append(axis.axisTag)
            if axis.axisTag not in self.state.current_location:
                self.state.current_location[axis.axisTag] = -1.0
        
        debug_log(f"populate_axis_data(): Result: {self.state.axis_min_max}")
        
        # Set default tool axis assignments if not already set
        # First axis -> horizontal (X), second axis -> vertical (Y)
        if self.state.tool_horizontal_axis is None and len(axis_tags) >= 1:
            self.state.tool_horizontal_axis = axis_tags[0]
            debug_log(f"populate_axis_data(): Default horizontal axis set to '{axis_tags[0]}'")
        if self.state.tool_vertical_axis is None and len(axis_tags) >= 2:
            self.state.tool_vertical_axis = axis_tags[1]
            debug_log(f"populate_axis_data(): Default vertical axis set to '{axis_tags[1]}'")
        
        # Track axis count for UI rebuild detection
        self.state._last_axis_count = len(axes)

    @objc.python_method
    def setup_axes(self):
        """Build the palette UI with dynamic axis sliders."""
        debug_log(f"setup_axes(): Building palette UI for instance {id(self)}")
        
        # Store axis slider groups - will be populated dynamically
        self._axis_slots = []
        self._tool_axis_tags = [None]
        self._current_axis_count = 0
        
        # Build the initial UI (no axis sliders yet - will be added in rebuild_axis_sliders)
        self._build_initial_ui()
    
    @objc.python_method
    def _clear_group_controls(self):
        """Remove dynamic controls from the palette group."""
        group = self.paletteView.group

        # Remove axis slots
        for i in range(len(getattr(self, '_axis_slots', []))):
            attr_name = f"axis_slot{i}"
            if hasattr(group, attr_name):
                delattr(group, attr_name)
        self._axis_slots = []

        # Remove known controls
        control_names = [
            'fillCheck', 'nodesCheck', 'anchorsCheck',
            'previewWindowCheck', 'previewBubbleCheck',
            'bubbleSizeSlider', 'bubbleSizeValue', 'kinksCheck',
            'toolMappingLabel', 'horizontalAxisLabel', 'horizontalAxisPopup',
            'verticalAxisLabel', 'verticalAxisPopup', 'globalPlayButton'
        ]
        for name in control_names:
            if hasattr(group, name):
                delattr(group, name)

    @objc.python_method
    def _build_ui(self, axes_data):
        """Build palette UI controls, optionally including axis sliders."""
        group = self.paletteView.group
        row_height = self._row_height
        spacing = self._spacing
        left_margin = self._left_margin

        y = 0
        self._axis_slots = []

        # Axis sliders
        for i, (axis, min_val, max_val) in enumerate(axes_data):
            axisgroup = AxisSlider(
                axis, min_val, max_val,
                posSize=(0, y, -0, self._axis_row_height),
                owner=self,
                callback=self.update_position
            )
            attr_name = f"axis_slot{i}"
            setattr(group, attr_name, axisgroup)
            self._axis_slots.append(axisgroup)
            anim = self.state.ensure_axis_animation(axis.axisTag)
            if axis.axisTag in self.state.current_location:
                current_actual = self.state.get_actual_value(axis.axisTag)
            else:
                current_actual = min_val
                self.state.set_from_actual_value(axis.axisTag, current_actual)
            axisgroup.slider.set(current_actual)
            axisgroup.valuebox.set(f"{current_actual:.1f}")
            axisgroup.set_play_state(anim.get("is_playing", False))
            axisgroup.set_speed_label(anim.get("speed", 1.0))
            y += self._axis_row_height + spacing

        # Global play/pause (all axes) row
        if axes_data:
            group.globalPlayButton = Button(
                (left_margin, y, 70, row_height + 4),
                "Play All",
                sizeStyle="mini",
                callback=self.togglePlayAll,
            )
            y += row_height + spacing
            self._refresh_global_play_button()

        # Tool axis mapping only if axes exist
        if axes_data:
            y += 5  # Small extra spacing before section

            group.toolMappingLabel = TextBox(
                (left_margin, y, -0, row_height),
                "Tool Axis Mapping:",
                sizeStyle="mini"
            )
            y += row_height

            axis_options = ["None"] + [f"{axis.name} ({axis.axisTag})" for axis, _, _ in axes_data]
            self._tool_axis_tags = [None] + [axis.axisTag for axis, _, _ in axes_data]

            # Horizontal (X) axis dropdown
            group.horizontalAxisLabel = TextBox(
                (left_margin, y, 35, row_height),
                "X axis:",
                sizeStyle="mini"
            )

            h_axis_index = 0
            if self.state.tool_horizontal_axis and self.state.tool_horizontal_axis in self._tool_axis_tags:
                h_axis_index = self._tool_axis_tags.index(self.state.tool_horizontal_axis)

            group.horizontalAxisPopup = PopUpButton(
                (40, y - 2, -5, row_height + 4),
                axis_options,
                callback=self.horizontalAxisChanged,
                sizeStyle="mini"
            )
            group.horizontalAxisPopup.set(h_axis_index)
            y += row_height + spacing

            # Vertical (Y) axis dropdown
            group.verticalAxisLabel = TextBox(
                (left_margin, y, 35, row_height),
                "Y axis:",
                sizeStyle="mini"
            )

            v_axis_index = 0
            if self.state.tool_vertical_axis and self.state.tool_vertical_axis in self._tool_axis_tags:
                v_axis_index = self._tool_axis_tags.index(self.state.tool_vertical_axis)

            group.verticalAxisPopup = PopUpButton(
                (40, y - 2, -5, row_height + 4),
                axis_options,
                callback=self.verticalAxisChanged,
                sizeStyle="mini"
            )
            group.verticalAxisPopup.set(v_axis_index)
            y += row_height + spacing

        # Track axis count and resize palette
        self._current_axis_count = len(axes_data)
        self._update_palette_height(y+5)

    @objc.python_method
    def _build_initial_ui(self):
        """Build the initial palette UI structure."""
        # Create group with flexible height - will be resized dynamically
        # Start large, then resize to fit content
        self.paletteView.group = Group((0, 0, 150, 400))
        
        # Layout parameters
        self._row_height = 14
        self._axis_row_height = 34
        self._spacing = 2
        self._left_margin = 5
        
        # Axis slots will be populated by rebuild_axis_sliders()
        self._axis_slots = []
        self._current_axis_count = 0

        # Build shared UI with no axes yet
        self._build_ui([])
    
    @objc.python_method
    def _rebuild_ui_content(self, num_axes, axes_data):
        """Rebuild the entire UI content with the given number of axis sliders."""
        self._clear_group_controls()
        self._build_ui(axes_data)
    
    @objc.python_method
    def _update_palette_height(self, height):
        """Update the palette height using the NSLayoutConstraint."""
        try:
            # Store the current height for minHeight() method
            self._current_height = height
            
            # Update the height constraint - this is the proper way to resize Glyphs palettes
            if hasattr(self, 'heightConstraint') and self.heightConstraint:
                self.heightConstraint.setConstant_(height)
            
        except Exception as e:
            debug_log(f"_update_palette_height failed: {e}")
    
    @objc.python_method
    def rebuild_axis_sliders(self):
        """
        Rebuild axis sliders when switching to a font with different axes.
        Creates/removes sliders dynamically and resizes palette.
        """
        # Use _windowController to get the correct font for THIS palette
        font = None
        if hasattr(self, '_windowController') and self._windowController:
            font = self._windowController.documentFont()
        if not font:
            font = Glyphs.font  # Fallback (but should not happen after first update)
            
        if not font or not font.axes:
            return
        
        current_axes = list(font.axes)
        num_axes = len(current_axes)
        
        # Check if we need to rebuild (axis count changed)
        current_count = getattr(self, '_current_axis_count', 0)
        rebuild_needed = (num_axes != current_count)
        
        if rebuild_needed:
            debug_log(f"rebuild_axis_sliders(): Rebuilding for {num_axes} axes (was {current_count})")
            
            # Build axis data list
            axes_data = []
            for axis in current_axes:
                min_value, max_value = self.state.axis_min_max.get(axis.axisTag, (0, 1000))
                axes_data.append((axis, min_value, max_value))
            
            # Rebuild the entire UI with new axes
            self._rebuild_ui_content(num_axes, axes_data)
        else:
            # Just update existing axis sliders with current font data
            for slot_idx, slot_group in enumerate(self._axis_slots):
                if slot_idx < len(current_axes):
                    axis = current_axes[slot_idx]
                    min_value, max_value = self.state.axis_min_max.get(axis.axisTag, (0, 1000))
                    
                    # Update the slot's axis reference and values
                    slot_group.axis = axis
                    
                    # Update the label text
                    slot_group.label.set(axis.axisTag)
                    
                    # Access the underlying NSSlider to set min/max
                    ns_slider = slot_group.slider.getNSSlider()
                    ns_slider.setMinValue_(min_value)
                    ns_slider.setMaxValue_(max_value)
                    current_actual = self.state.get_actual_value(axis.axisTag)
                    slot_group.slider.set(current_actual)
                    slot_group.valuebox.set(f"{current_actual:.1f}")
                    anim = self.state.ensure_axis_animation(axis.axisTag)
                    slot_group.set_play_state(anim.get("is_playing", False))
                    slot_group.set_speed_label(anim.get("speed", 1.0))

    @objc.python_method
    def update_position(self, axis, value):
        
        axis_tag = axis.axisTag
        # Safety check: ensure axis data exists (may be cleared on font switch)
        if axis_tag not in self.state.axis_min_max:
            debug_log(f"  -> Axis {axis_tag} NOT in axis_min_max! Repopulating...")
            # Try to repopulate axis data
            self.populate_axis_data()
            if axis_tag not in self.state.axis_min_max:
                # Axis doesn't exist in current font, ignore
                debug_log(f"  -> Axis {axis_tag} STILL not in axis_min_max after repopulate!")
                return
        
        min_val, max_val = self.state.axis_min_max[axis_tag]
        self.state.current_location[axis_tag] = normalizeValue(
            value,
            (min_val, min_val, max_val),
        )
        debug_log(f"  -> Normalized: current_location[{axis_tag}] = {self.state.current_location[axis_tag]}")
        if self.state.model:
            self.state.master_scalars = self.state.model.getMasterScalars(self.state.current_location)
            debug_log(f"  -> Updated master_scalars, count={len(self.state.master_scalars)}")
        else:
            debug_log(f"  -> No model! model={self.state.model}")
        self.state.update_glyph()
        debug_log(f"  -> Called update_glyph()")

    @objc.python_method
    def _ensure_animation_timer(self):
        if getattr(self, '_animation_timer', None):
            return
        try:
            self._animation_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.016, self, objc.selector(self.animationTick_, signature=b"v@:@"), None, True
            )
        except Exception:
            self._animation_timer = None

    @objc.python_method
    def _invalidate_animation_timer(self):
        try:
            if getattr(self, '_animation_timer', None):
                self._animation_timer.invalidate()
        except Exception:
            pass
        self._animation_timer = None

    @objc.typedSelector(b'v@:@')
    def animationTick_(self, timer):
        # Safety check - don't run if state is gone
        if not hasattr(self, 'state') or not self.state or not self.state.axis_min_max:
            return
        
        now = time.time()

        updates = {}
        for axis_tag, (min_val, max_val) in self.state.axis_min_max.items():
            anim = self.state.ensure_axis_animation(axis_tag)

            # Pause while scrubbing, resume after pause window
            paused_until = anim.get("paused_until")
            if paused_until is not None and now < paused_until:
                anim["last_time"] = now
                continue
            if paused_until is not None and now >= paused_until:
                anim["paused_until"] = None
                if anim.pop("resume_after_scrub", False):
                    anim["is_playing"] = True
                anim["last_time"] = now

            if not anim.get("is_playing", False):
                anim["last_time"] = now
                continue

            last_time = anim.get("last_time")
            if last_time is None:
                anim["last_time"] = now
                continue

            delta = now - last_time
            anim["last_time"] = now
            if delta <= 0:
                continue

            speed = max(anim.get("speed", 1.0), 0.01)
            duration = 1.0 / speed  # seconds per leg (min->max)
            span = max_val - min_val
            if span == 0:
                continue

            direction = anim.get("direction", 1)
            step = span * (delta / duration) * direction

            current_actual = self.state.get_actual_value(axis_tag)
            new_val = current_actual + step

            # Ping-pong at bounds
            while True:
                if new_val > max_val:
                    excess = new_val - max_val
                    new_val = max_val - excess
                    direction = -1
                elif new_val < min_val:
                    excess = min_val - new_val
                    new_val = min_val + excess
                    direction = 1
                else:
                    break

            anim["direction"] = direction
            updates[axis_tag] = new_val

        if updates:
            self.state.apply_axis_updates(updates)
            self._sync_axis_controls(updates)
            Glyphs.redraw()

    @objc.python_method
    def _sync_axis_controls(self, updates: Dict[str, float]):
        if not updates:
            return
        for slot in getattr(self, '_axis_slots', []):
            if not hasattr(slot, 'axis') or slot.axis is None:
                continue
            tag = slot.axis.axisTag
            if tag in updates:
                val = updates[tag]
                slot.slider.set(val)
                slot.valuebox.set(f"{val:.1f}")

    @objc.python_method
    def _set_axis_control_value(self, axis_tag: str, value: float):
        for slot in getattr(self, '_axis_slots', []):
            if hasattr(slot, 'axis') and slot.axis and slot.axis.axisTag == axis_tag:
                slot.slider.set(value)
                slot.valuebox.set(f"{value:.1f}")
                break

    @objc.python_method
    def _set_axis_play_state(self, axis_tag: str, is_playing: bool):
        for slot in getattr(self, '_axis_slots', []):
            if hasattr(slot, 'axis') and slot.axis and slot.axis.axisTag == axis_tag:
                slot.set_play_state(is_playing)
                break

    @objc.python_method
    def _set_axis_speed_label(self, axis_tag: str, speed: float):
        for slot in getattr(self, '_axis_slots', []):
            if hasattr(slot, 'axis') and slot.axis and slot.axis.axisTag == axis_tag:
                slot.set_speed_label(speed)
                break

    @objc.python_method
    def _refresh_global_play_button(self):
        try:
            group = self.paletteView.group
            if hasattr(group, 'globalPlayButton'):
                any_playing = any(a.get("is_playing", False) for a in self.state.axis_animation.values()) if self.state.axis_animation else False
                title = "Pause All" if any_playing else "Play All"
                group.globalPlayButton.getNSButton().setTitle_(title)
        except Exception:
            pass

    @objc.python_method
    def on_axis_scrub(self, axis):
        tag = axis.axisTag
        anim = self.state.ensure_axis_animation(tag)
        anim["paused_until"] = time.time() + 0.25
        anim["resume_after_scrub"] = anim.get("is_playing", False)
        anim["is_playing"] = False
        anim["last_time"] = None

    @objc.python_method
    def on_axis_play_toggle(self, axis, is_double: bool):
        tag = axis.axisTag
        anim = self.state.ensure_axis_animation(tag)
        now = time.time()

        if is_double:
            # Stop and reset to min
            anim.update({
                "is_playing": False,
                "direction": 1,
                "last_time": None,
                "paused_until": None,
                "resume_after_scrub": False,
            })
            min_val, _ = self.state.axis_min_max.get(tag, (0, 0))
            self.state.apply_axis_updates({tag: min_val})
            self._sync_axis_controls({tag: min_val})
            self._set_axis_play_state(tag, False)
            self._refresh_global_play_button()
            return

        new_state = not anim.get("is_playing", False)
        anim["is_playing"] = new_state
        anim["last_time"] = now
        anim["paused_until"] = None
        self._set_axis_play_state(tag, new_state)
        self._refresh_global_play_button()

    @objc.python_method
    def on_axis_speed_change(self, axis, speed: float):
        tag = axis.axisTag
        anim = self.state.ensure_axis_animation(tag)
        anim["speed"] = speed
        anim["last_time"] = time.time()
        self._set_axis_speed_label(tag, speed)

    @objc.python_method
    def togglePlayAll(self, sender):
        is_double = False
        try:
            evt = NSApplication.sharedApplication().currentEvent()
            if evt and hasattr(evt, 'clickCount'):
                is_double = evt.clickCount() >= 2
        except Exception:
            pass

        now = time.time()
        any_playing = any(a.get("is_playing", False) for a in self.state.axis_animation.values()) if self.state.axis_animation else False

        if is_double:
            # Stop and reset all
            updates = {}
            for tag, (min_val, _) in self.state.axis_min_max.items():
                anim = self.state.ensure_axis_animation(tag)
                anim.update({
                    "is_playing": False,
                    "direction": 1,
                    "last_time": None,
                    "paused_until": None,
                    "resume_after_scrub": False,
                })
                updates[tag] = min_val
            if updates:
                self.state.apply_axis_updates(updates)
                self._sync_axis_controls(updates)
            self._refresh_global_play_button()
            return

        target_state = not any_playing
        for tag in self.state.axis_min_max.keys():
            anim = self.state.ensure_axis_animation(tag)
            anim["is_playing"] = target_state
            anim["paused_until"] = None
            anim["resume_after_scrub"] = False
            anim["last_time"] = now
        self._refresh_global_play_button()

    @objc.python_method
    def horizontalAxisChanged(self, sender):
        """Callback when horizontal (X) axis popup changes."""
        # Safety check - state may be invalid during shutdown
        if not hasattr(self, 'state') or self.state is None:
            return
        try:
            index = sender.get()
            tool_axis_tags = getattr(self, '_tool_axis_tags', [])
            if index < len(tool_axis_tags):
                self.state.tool_horizontal_axis = tool_axis_tags[index]
                debug_log(f"Horizontal axis changed to: {self.state.tool_horizontal_axis}")
        except Exception:
            pass

    @objc.python_method
    def verticalAxisChanged(self, sender):
        """Callback when vertical (Y) axis popup changes."""
        # Safety check - state may be invalid during shutdown
        if not hasattr(self, 'state') or self.state is None:
            return
        try:
            index = sender.get()
            tool_axis_tags = getattr(self, '_tool_axis_tags', [])
            if index < len(tool_axis_tags):
                self.state.tool_vertical_axis = tool_axis_tags[index]
                debug_log(f"Vertical axis changed to: {self.state.tool_vertical_axis}")
        except Exception:
            pass

    @objc.python_method
    def openPreviewWindow(self):
        """Create and show the preview window"""
        # Safety check - state may be invalid during shutdown
        if not hasattr(self, 'state') or self.state is None:
            return
        
        if self.state.preview_window:
            self.state.preview_window.makeKeyAndOrderFront_(None)
            return
        
        # Create window - only titled and resizable, no close/miniaturize buttons
        # User must toggle via the palette checkbox
        windowRect = NSMakeRect(100, 100, 800, 300)
        styleMask = NSTitledWindowMask | NSResizableWindowMask
        
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            windowRect,
            styleMask,
            NSBackingStoreBuffered,
            False
        )
        window.setTitle_("Interpolation Preview")
        window.setMinSize_((400, 150))
        
        # Create custom view
        contentRect = window.contentView().bounds()
        previewView = InterpolatePreviewView.alloc().initWithFrame_(contentRect)
        previewView.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        previewView.setInterpolateState_(self.state)
        
        window.contentView().addSubview_(previewView)
        # Don't set delegate - window can only be closed via palette checkbox
        
        self.state.preview_window = window
        self.state.preview_view = previewView
        
        window.makeKeyAndOrderFront_(None)
        
        # Initial update
        self.state.update_preview_window()
    
    @objc.python_method
    def closePreviewWindow(self):
        """Close the preview window safely"""
        try:
            # Get references
            window = getattr(self.state, 'preview_window', None) if self.state else None
            view = getattr(self.state, 'preview_view', None) if self.state else None
            
            # Clear state references FIRST to prevent any callbacks during close
            if self.state:
                self.state.preview_window = None
                self.state.preview_view = None
            
            # Clear the view's state reference to prevent drawing during close
            if view:
                try:
                    view.state = None
                except Exception:
                    pass
            
            # Now close the window
            if window:
                try:
                    window.orderOut_(None)
                except Exception:
                    pass
        except Exception:
            # Ensure state is clean even if close fails
            if self.state:
                self.state.preview_window = None
                self.state.preview_view = None

    @objc.python_method
    def start(self):
        # start() is called when palette becomes active
        # UI is already built in settings(), just add callback
        Glyphs.addCallback(self.update, UPDATEINTERFACE)
        debug_log(f"Palette start() called - added UPDATEINTERFACE callback")
        self._ensure_animation_timer()
    
    @objc.python_method
    def _cleanup(self):
        """
        Proper cleanup method - more reliable than __del__.
        Called when palette is being destroyed or font is closing.
        """
        try:
            # Stop animation timer first
            self._invalidate_animation_timer()
        except Exception:
            pass
        
        try:
            # Remove callback
            Glyphs.removeCallback(self.update)
        except Exception:
            pass
        
        try:
            # Close preview window if open
            if hasattr(self, 'state') and self.state:
                self.closePreviewWindow()
        except Exception:
            pass
        
        try:
            # Unregister state from registry
            if hasattr(self, '_instance_font_sig') and self._instance_font_sig:
                if self._instance_font_sig in InterpolateState._instances:
                    del InterpolateState._instances[self._instance_font_sig]
        except Exception:
            pass
        
        try:
            # Clear state reference
            if hasattr(self, 'state'):
                self.state = None
        except Exception:
            pass

    @objc.python_method
    def __del__(self):
        # __del__ is unreliable but we keep it as a last-resort fallback
        self._cleanup()

    @objc.python_method
    def update(self, sender):
        # Safety check - don't run if state is gone (during cleanup)
        if not hasattr(self, 'state') or self.state is None:
            return
        
        # Check if Reporter is active - if not active recently, skip all work
        # This ensures the plugin does nothing when turned off from the View menu
        # Allow 0.5 second grace period for redraw timing
        if not InterpolateState._reporter_active:
            return
        now = time.time()  # Cache time.time() to avoid multiple syscalls
        if now - InterpolateState._reporter_last_active_time > 0.5:
            # Reporter hasn't been active in 0.5 seconds - mark as inactive and skip
            InterpolateState._reporter_active = False
            return
        
        currentTab = sender.object()
        if not isinstance(currentTab, GSEditViewController):
            return
        
        # Use this palette instance's window controller to get the correct font
        # This ensures each palette only responds to its own window's font
        if not self._windowController:
            return
        
        font = self._windowController.documentFont()
        if not font or not font.selectedLayers or len(font.masters) == 1:
            return
        
        font_name = getattr(font, 'familyName', 'Unknown')
        selected_layers = font.selectedLayers
        glyph_name = selected_layers[0].parent.name if selected_layers[0].parent else 'Unknown'
        
        # Build a font signature for THIS palette instance
        new_font_sig = InterpolateState._get_font_signature_static(font)
        
        # Debug: Log every update call to understand flow
        debug_log(f"update() palette={id(self)}, font='{font_name}', state={id(self.state)}, sig={new_font_sig}")
        debug_log(f"  -> state.axis_min_max keys: {list(self.state.axis_min_max.keys()) if self.state.axis_min_max else []}")
        debug_log(f"  -> state.model: {self.state.model is not None}, master_scalars: {len(self.state.master_scalars) if self.state.master_scalars else 0}")
        debug_log(f"  -> _initialized_for_font: {getattr(self, '_initialized_for_font', False)}, _instance_font_sig: {getattr(self, '_instance_font_sig', None)}")
        
        # Check if THIS palette instance needs initial setup (first update after settings())
        # This is when we have _windowController and can properly identify the font
        if not getattr(self, '_initialized_for_font', False):
            debug_log(f"  -> FIRST INITIALIZATION for this palette with font '{font_name}'")
            self._instance_font_sig = new_font_sig
            self._initialized_for_font = True
            
            # NOW populate axis data with the CORRECT font from _windowController
            self.populate_axis_data()
            debug_log(f"  -> axis_min_max after initial populate: {self.state.axis_min_max}")
            
            # Rebuild axis sliders to show correct axes
            self.rebuild_axis_sliders()
            
            # Build the model for this font
            self.state.build_model()
            debug_log(f"  -> model built, master_scalars={len(self.state.master_scalars) if self.state.master_scalars else 0}")
            
            # Register this state with the font (so Reporter can find it)
            InterpolateState.register_state(font, self.state)
            # Store palette reference in state so tool can sync sliders
            self.state._palette = self
            debug_log(f"  -> State registered for font '{font_name}'")
            
            self.state.current_glyph = None  # Force glyph setup
        else:
            # Check if THIS instance's font changed (e.g., font swapped in window)
            old_font_sig = getattr(self, '_instance_font_sig', None)
            font_changed = (new_font_sig != old_font_sig)
            
            if font_changed:
                debug_log(f"  -> FONT CHANGED! (old={old_font_sig})")
                self._instance_font_sig = new_font_sig
                
                # Register this state with the new font (so Reporter can find it)
                InterpolateState.register_state(font, self.state)
                # Store palette reference in state so tool can sync sliders
                self.state._palette = self
                
                # Repopulate axis data for new font
                self.populate_axis_data()
                debug_log(f"  -> axis_min_max after populate: {self.state.axis_min_max}")
                
                # Rebuild axis sliders for THIS palette instance
                self.rebuild_axis_sliders()
                
                # IMPORTANT: Invalidate the old model so it gets rebuilt for the new font
                self.state.model = None
                self.state.master_scalars = []
                self.state.build_model()
                debug_log(f"  -> model rebuilt, master_scalars={len(self.state.master_scalars) if self.state.master_scalars else 0}")
                self.state.current_glyph = None  # Force glyph rebuild
        
        # Check if master changed (one-shot snap when follow_masters is on)
        try:
            current_master = selected_layers[0].master if selected_layers else None
            current_master_id = getattr(current_master, 'id', None)
            if current_master_id != getattr(self.state, '_last_master_id', None):
                self.state._last_master_id = current_master_id
                if getattr(self.state, 'follow_masters', False) and current_master:
                    applied = self.state.set_location_from_master(current_master, font.axes)
                    if applied:
                        # Sync UI controls to the master's position
                        try:
                            updates = {}
                            for axis in font.axes:
                                if axis.axisTag in self.state.current_location:
                                    updates[axis.axisTag] = self.state.get_actual_value(axis.axisTag)
                            self._sync_axis_controls(updates)
                        except Exception:
                            pass
                        Glyphs.redraw()
        except Exception as e:
            debug_log(f"update(): master change handling error: {e}")

        # Check if glyph changed
        current_glyph = selected_layers[0].parent
        if (
            self.state.current_glyph is None
            or current_glyph != self.state.current_glyph
        ):
            debug_log(f"update(): Glyph changed to '{glyph_name}'")
            self.state.glyph_points = {}
            self.state.clear_caches(clear_kinks=True)  # Clear all caches including kinks for new glyph
            
            # Invalidate any stale sync preview when glyph changes
            SyncRatiosPanel.invalidate_preview_if_stale(current_glyph, font)
            
            # Ensure axes data is set up before building model (might be empty after clear_all_caches)
            if not self.state.axis_min_max:
                debug_log("  -> axis_min_max is empty, repopulating")
                self.populate_axis_data()
            
            self.state.build_model()
            self.state.current_glyph = current_glyph
            
            # Detect potential kinks for new glyph if kinks are enabled
            if self.state.show_kinks:
                self.state._detect_potential_kinks()
        else:
            # Even if glyph didn't change, check if content changed (undo/redo/edit)
            # This ensures stale previews are cleared after undo
            SyncRatiosPanel.invalidate_preview_if_stale(current_glyph, font)
        
        # Debug state before update_glyph
        debug_log(f"update(): Calling update_glyph() - model={self.state.model is not None}, master_scalars={len(self.state.master_scalars) if self.state.master_scalars else 0}, axis_min_max keys={list(self.state.axis_min_max.keys())}")
        self.state.update_glyph()

    @objc.python_method
    def __file__(self):
        """Please leave this method unchanged"""
        return __file__


class InterpolateTool(SelectTool):
    """
    A tool that allows mouse-drag control of axis values.
    
    When this tool is selected from the toolbar:
    - The Reporter is automatically enabled if it was off
    - Horizontal mouse movement controls the first assigned axis
    - Vertical mouse movement controls the second assigned axis
    - Axes can be assigned via ComboBoxes in the Palette
    - Selection/editing is completely disabled - tool is inert except for axis control
    """
    
    @objc.python_method
    def settings(self):
        self.name = Glyphs.localize({
            'en': 'Interpol Tool'
        })
        
        # Keyboard shortcut - load from preferences or use default
        self.keyboardShortcut = InterpolateConfig.get_tool_shortcut()
        
        # Toolbar position - gives this tool its own slot in the toolbar
        # Must be different from other tools (Overlapper uses 100)
        self.toolbarPosition = 150
        
        # Load toolbar icon from PDF file (like the SelectTool template)
        icon_path = os.path.join(os.path.dirname(self.__file__()), "toolbarIconTemplate.pdf")
        self.tool_bar_image = NSImage.alloc().initByReferencingFile_(icon_path)
        self._icon = None  # Needs to be set to None for now (fixed in Glyphs 3.4)
        
        # Context menu items
        self.generalContextMenus = [
            {
                'name': Glyphs.localize({
                    'en': 'Reset to center'
                }),
                'action': self.resetAxesToCenter_
            },
        ]
        
        # Tracking state for drag operation
        self._drag_start_location = None
        self._drag_start_axis_values = {}
        self._current_mouse_location = None  # For drawing cursor tooltip
    
    @objc.python_method
    def start(self):
        """Called when plugin is loaded"""
        pass
    
    @objc.python_method
    def activate(self):
        """Called when tool is selected from the toolbar"""
        debug_log("InterpolateTool: activate()")
        
        # Turn on the Reporter if it's off
        self._ensure_reporter_active()
    
    @objc.python_method
    def _ensure_reporter_active(self):
        """Ensure the Interpolate Reporter is active."""
        try:
            font = Glyphs.font
            if not font:
                return
            
            # Find the InterpolateReporter in registered reporters
            for reporter in Glyphs.reporters:
                if reporter.__class__.__name__ == 'InterpolateReporter':
                    # Check if this reporter is active by checking activeReporters
                    # The reporter should toggle itself via its View menu item
                    try:
                        # Try to toggle via the reporter's toggle method
                        # If the reporter is not active, this will activate it
                        # Check if reporter's show_preview state indicates it should be on
                        state = InterpolateState.get_state_for_font(font)
                        if state and not state.show_preview:
                            state.show_preview = True
                            state.save_defaults()
                            Glyphs.redraw()
                    except Exception:
                        pass
                    break
        except Exception as e:
            debug_log(f"InterpolateTool._ensure_reporter_active() error: {e}")
    
    @objc.python_method
    def deactivate(self):
        """Called when switching away from this tool"""
        debug_log("InterpolateTool: deactivate()")
        self._drag_start_location = None
        self._drag_start_axis_values = {}
    
    @objc.python_method
    def _get_state(self):
        """Get the InterpolateState for the current font."""
        font = Glyphs.font
        if not font:
            return None
        return InterpolateState.get_state_for_font(font)
    
    def mouseDown_(self, theEvent):
        """Handle mouse down - record starting position and axis values.
        NOTE: We do NOT call super().mouseDown_() to prevent any selection behavior.
        """
        # Let double-clicks behave like the default SelectTool (e.g. switching glyphs)
        try:
            if hasattr(theEvent, "clickCount") and theEvent.clickCount() >= 2:
                objc.super(InterpolateTool, self).mouseDown_(theEvent)
                return
        except Exception as e:
            debug_log(f"InterpolateTool.mouseDown_ double-click fallback error: {e}")
        
        try:
            state = self._get_state()
            if not state:
                debug_log("InterpolateTool.mouseDown_: No state found")
                return
            
            # Get the mouse location in view coordinates
            loc = self.editViewController().graphicView().getActiveLocation_(theEvent)
            self._drag_start_location = (loc.x, loc.y)
            self._current_mouse_location = (loc.x, loc.y)
            
            # Store starting axis values
            self._drag_start_axis_values = dict(state.current_location)
            
            # Trigger redraw to show cursor tooltip
            Glyphs.redraw()
            
            debug_log(f"InterpolateTool.mouseDown_: start={self._drag_start_location}, axes={self._drag_start_axis_values}")
        except Exception as e:
            debug_log(f"InterpolateTool.mouseDown_ error: {e}")
    
    def mouseDragged_(self, theEvent):
        """Handle mouse drag - update axis values based on movement.
        NOTE: We do NOT call super().mouseDragged_() to prevent any selection behavior.
        """
        # objc.super(InterpolateTool, self).mouseDragged_(theEvent)
        
        try:
            if self._drag_start_location is None:
                return
            
            state = self._get_state()
            if not state:
                return
            
            # Debug: show which state the tool is using
            debug_log_lazy(lambda state=state: f"InterpolateTool.mouseDragged_: Using state id={id(state)}")
            
            # Get current location
            loc = self.editViewController().graphicView().getActiveLocation_(theEvent)
            current_location = (loc.x, loc.y)
            self._current_mouse_location = current_location  # Store for cursor tooltip
            
            # Calculate delta from start (in view coordinates, unscaled)
            dx = current_location[0] - self._drag_start_location[0]
            dy = current_location[1] - self._drag_start_location[1]
            
            # Get view scale to convert to screen pixels
            view_scale = self.editViewController().graphicView().scale()
            
            # Convert to normalized axis delta
            # Sensitivity: pixels of mouse movement per full axis traverse (-1 to 1 = 2 units)
            axis_dx = (dx * view_scale) / InterpolateConfig.TOOL_MOUSE_SENSITIVITY
            axis_dy = (dy * view_scale) / InterpolateConfig.TOOL_MOUSE_SENSITIVITY
            
            # Track if any axis changed
            changed = False
            
            # Update horizontal axis if assigned
            h_axis = state.tool_horizontal_axis
            if h_axis and h_axis in state.axis_min_max:
                start_val = self._drag_start_axis_values.get(h_axis, 0)
                new_val = start_val + axis_dx
                # Clamp to normalized range (0 to 1, since default=min in normalizeValue)
                new_val = max(0.0, min(1.0, new_val))
                state.current_location[h_axis] = new_val
                debug_log_lazy(lambda h=h_axis, n=new_val, a=state.get_actual_value(h_axis): f"InterpolateTool: h_axis={h}, norm={n:.3f}, actual={a:.1f}")
                changed = True
            
            # Update vertical axis if assigned
            v_axis = state.tool_vertical_axis
            if v_axis and v_axis in state.axis_min_max:
                start_val = self._drag_start_axis_values.get(v_axis, 0)
                new_val = start_val + axis_dy
                # Clamp to normalized range (0 to 1, since default=min in normalizeValue)
                new_val = max(0.0, min(1.0, new_val))
                state.current_location[v_axis] = new_val
                debug_log_lazy(lambda v=v_axis, n=new_val, a=state.get_actual_value(v_axis): f"InterpolateTool: v_axis={v}, norm={n:.3f}, actual={a:.1f}")
                changed = True
            
            if changed:
                # Recalculate master scalars based on new location
                if state.model:
                    state.master_scalars = state.model.getMasterScalars(state.current_location)
                
                # Clear caches and trigger redraw
                state.clear_caches()
                state.update_glyph()
                Glyphs.redraw()
            
        except Exception as e:
            debug_log(f"InterpolateTool.mouseDragged_ error: {e}")
    
    def mouseUp_(self, theEvent):
        """Handle mouse up - finalize the drag operation.
        NOTE: We do NOT call super().mouseUp_() to prevent any selection behavior.
        """
        # Intentionally NOT calling super - we don't want any selection behavior
        # objc.super(InterpolateTool, self).mouseUp_(theEvent)
        
        try:
            # Sync palette sliders with final axis values
            self._sync_palette_sliders()
            
            self._drag_start_location = None
            self._drag_start_axis_values = {}
            self._current_mouse_location = None  # Hide cursor tooltip
            
            # Final redraw to clear tooltip
            Glyphs.redraw()
            
            debug_log("InterpolateTool.mouseUp_: drag complete")
        except Exception as e:
            debug_log(f"InterpolateTool.mouseUp_ error: {e}")
    
    @objc.python_method
    def _sync_palette_sliders(self):
        """Sync the palette sliders with the current axis values."""
        try:
            state = self._get_state()
            if not state:
                debug_log("InterpolateTool._sync_palette_sliders: No state found")
                return
            
            # Get the palette directly from the state (stored when palette registers)
            palette = getattr(state, '_palette', None)
            if not palette:
                debug_log("InterpolateTool._sync_palette_sliders: No palette reference in state")
                return
            
            # Update the palette sliders
            self._update_palette_sliders(palette, state)
            debug_log("InterpolateTool._sync_palette_sliders: Updated palette sliders")
            
        except Exception as e:
            debug_log(f"InterpolateTool._sync_palette_sliders error: {e}")
    
    @objc.python_method
    def _update_palette_sliders(self, palette, state):
        """Update the slider values in the palette to match current state."""
        try:
            if not hasattr(palette, '_axis_slots'):
                return
            
            for slot in palette._axis_slots:
                axis_tag = slot.axis.axisTag
                if axis_tag in state.current_location and axis_tag in state.axis_min_max:
                    actual_val = state.get_actual_value(axis_tag)
                    # Update slider and value box without triggering callback
                    slot.slider.set(actual_val)
                    slot.valuebox.set(str(int(actual_val)))
        except Exception as e:
            debug_log(f"InterpolateTool._update_palette_sliders error: {e}")
    
    def mouseMoved_(self, theEvent):
        """Handle mouse move - we don't need to do anything special here."""
        # Don't call super to avoid any hover effects
        pass
    
    def selectAllPathsAndSelectAllNodes_(self, sender):
        """Override to prevent select all behavior."""
        pass
    
    def selectAll_(self, sender):
        """Override to prevent select all behavior."""
        pass
    
    def doubleClickAtPoint_OnNode_OnPath_OnAnchor_OnGuideLine_OnComponent_OnText_(
        self, point, node, path, anchor, guideLine, component, text
    ):
        """Delegate double-click to SelectTool so glyph switching works."""
        try:
            objc.super(InterpolateTool, self).doubleClickAtPoint_OnNode_OnPath_OnAnchor_OnGuideLine_OnComponent_OnText_(
                point, node, path, anchor, guideLine, component, text
            )
        except Exception as e:
            debug_log(f"InterpolateTool.doubleClick error: {e}")
    
    def willSelectTempTool_(self, sender):
        """Allow temporary tool selection (e.g., holding space for the hand tool)."""
        try:
            return objc.super(InterpolateTool, self).willSelectTempTool_(sender)
        except Exception:
            return True
    
    @objc.python_method
    def foreground(self, layer):
        """Draw axis values near the cursor during drag."""
        if self._current_mouse_location is None:
            return
        
        state = self._get_state()
        if not state:
            return
        
        try:
            # Get scale factor
            try:
                scale = self.editViewController().graphicView().scale()
            except Exception:
                scale = 1.0
            
            # Build text showing current axis positions
            lines = []
            
            h_axis = state.tool_horizontal_axis
            if h_axis and h_axis in state.axis_min_max and h_axis in state.current_location:
                lines.append(f"X: {h_axis} = {int(state.get_actual_value(h_axis))}")
            
            v_axis = state.tool_vertical_axis
            if v_axis and v_axis in state.axis_min_max and v_axis in state.current_location:
                lines.append(f"Y: {v_axis} = {int(state.get_actual_value(v_axis))}")
            
            if not lines:
                return
            
            # Position tooltip to the right and slightly above cursor
            x = self._current_mouse_location[0] + 20 / scale
            y = self._current_mouse_location[1] + 10 / scale
            
            DrawingHelpers.draw_tooltip((x, y), lines, scale)
            
        except Exception as e:
            debug_log(f"InterpolateTool.foreground error: {e}")
    
    def resetAxesToCenter_(self, sender):
        """Reset all axes to their center values (0 in normalized space)."""
        try:
            state = self._get_state()
            if not state:
                return
            
            # Reset all axes to center of normalized range (0)
            for axis_tag in state.axis_min_max.keys():
                state.current_location[axis_tag] = 0.0
            
            # Recalculate master scalars
            if state.model:
                state.master_scalars = state.model.getMasterScalars(state.current_location)
            
            state.clear_caches()
            state.update_glyph()
            Glyphs.redraw()
            
            debug_log("InterpolateTool.resetAxesToCenter_: Reset all axes to center")
        except Exception as e:
            debug_log(f"InterpolateTool.resetAxesToCenter_ error: {e}")
    
    @objc.python_method
    def __file__(self):
        """Please leave this method unchanged"""
        return __file__
