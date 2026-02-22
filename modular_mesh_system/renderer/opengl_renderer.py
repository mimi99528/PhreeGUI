"""
OpenGL渲染器模块
OpenGL Renderer Module
"""

import time
import math
import numpy as np
from typing import List, Optional

from OpenGL.GL import glGetString
import moderngl
from PIL import Image

from ..utils.config import CONFIG
from ..utils.helpers import (
    ThrottleLog, create_perspective_matrix, 
    create_lookat_matrix, calculate_camera_position
)

class OpenGLRenderer:
    """ModernGL OpenGL渲染器"""

    def __init__(self, mesh_system):
        self.mesh_system = mesh_system
        self.ctx = None
        self.prog = None
        self.vao = None
        self.fbo = None

        # 节流日志
        self._tlog = ThrottleLog(0.5)

        # 深度图保存相关变量
        self.last_depth_save_time = 0
        self.depth_save_interval = CONFIG.DEPTH_SAVE_INTERVAL
        self.depth_frame_counter = 0
        self.depth_save_enabled = CONFIG.ENABLE_DEPTH_TEXTURE_SAVE
        self.depth_save_path = CONFIG.DEPTH_SAVE_PATH
        self.depth_save_format = CONFIG.DEPTH_SAVE_FORMAT
        self.depth_normalize_range = CONFIG.DEPTH_NORMALIZE_RANGE
        
        # 创建保存目录
        import os
        if self.depth_save_enabled:
            os.makedirs(self.depth_save_path, exist_ok=True)

        # 顶点和索引计数
        self.vertex_count = 0
        self.index_count = 0

        # Camera parameters
        self.camera_distance = CONFIG.CAMERA_DISTANCE
        self.camera_angle_x = CONFIG.CAMERA_ANGLE_X
        self.camera_angle_y = CONFIG.CAMERA_ANGLE_Y
        self.camera_target = CONFIG.CAMERA_TARGET.copy()

        # Mesh data - 预分配缓冲区
        self.vertices = []
        self.triangles = []
        self.vertex_buffer = None
        self.index_buffer = None

        # 预分配的GPU缓冲区
        self.max_vertices = CONFIG.VERTEX_BUFFER_SIZE
        self.max_indices = CONFIG.INDEX_BUFFER_SIZE

        # 临时缓冲区用于数据准备
        self._temp_vertex_data = None
        self._temp_index_data = None
        self._temp_verts = None
        self._temp_normals = None
        self._temp_counts = None
        self._vertex_data = None

        # SSFR parameters
        self.use_ssfr = CONFIG.USE_SSFR
        self.ssfr_scale = CONFIG.SSFR_SCALE
        self.ssfr_filter_iterations = CONFIG.SSFR_FILTER_ITERATIONS
        self.ssfr_filter_intervals = CONFIG.SSFR_FILTER_INTERVALS
        self.ssfr_blur_sigma = CONFIG.SSFR_BLUR_SIGMA
        self.ssfr_depth_sigma = CONFIG.SSFR_DEPTH_SIGMA
        self.ssfr_blur_radius = CONFIG.SSFR_BLUR_RADIUS
        self.ssfr_particle_radius_world = CONFIG.SSFR_PARTICLE_RADIUS_WORLD

        # runtime size cache
        self._ssfr_w = 0
        self._ssfr_h = 0

        # particle GPU resources
        self.particle_vbo = None
        self.particle_vao_depth = None
        self.particle_vao_thick = None
        self.particle_depth_prog = None
        self.particle_thick_prog = None
        self.particle_count = 0

        # ssfr passes
        self.bilateral_prog = None
        self.quad_vao_bilateral = None
        self.quad_vao_comp = None
        self.quad_vbo = None
        self.composite_prog = None

        # ssfr textures/fbos
        self.depth_tex0 = None
        self.depth_tex1 = None
        self.thick_tex = None
        self.depth_fbo0 = None
        self.depth_fbo1 = None
        self.thick_fbo = None

        # Shadow mapping
        self.shadow_scale = 1.0
        self.shadow_w = 0
        self.shadow_h = 0
        self.shadow_tex = None
        self.shadow_fbo = None
        self.shadow_prog = None
        self.shadow_vao = None

        # light config
        self.light_pos = CONFIG.LIGHT_POSITION.copy()
        self.light_target = CONFIG.LIGHT_TARGET.copy()

    def initialize(self, ctx):
        """Initialize OpenGL context and shaders"""
        self.ctx = ctx

        # Shader code
        vertex_shader = '''
            #version 330
            uniform mat4 projection;
            uniform mat4 view;
            uniform mat4 model;

            layout(location = 0) in vec3 in_position;
            layout(location = 1) in vec3 in_normal;

            out vec3 v_normal_ws;
            out vec3 v_pos_ws;

            void main() {
                vec4 world_pos = model * vec4(in_position, 1.0);
                v_pos_ws = world_pos.xyz;
                mat3 nmat = mat3(transpose(inverse(model)));
                v_normal_ws = normalize(nmat * in_normal);
                gl_Position = projection * view * world_pos;
            }
        '''

        fragment_shader = '''
            #version 330
            uniform vec3 light_pos;
            uniform vec3 camera_pos;
            uniform vec3 water_color;
            uniform float alpha_base;
            uniform float fresnel_power;
            uniform float spec_power;
            uniform float absorption;

            in vec3 v_normal_ws;
            in vec3 v_pos_ws;

            out vec4 frag_color;

            void main() {
                vec3 N = normalize(v_normal_ws);
                vec3 V = normalize(camera_pos - v_pos_ws);
                vec3 L = normalize(light_pos - v_pos_ws);

                vec3 H = normalize(L + V);

                float NdotL = max(dot(N, L), 0.0);
                float NdotV = max(dot(N, V), 0.0);
                float spec = pow(max(dot(N, H), 0.0), spec_power);

                float F0 = 0.02;
                float fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, fresnel_power);

                float thickness = 1.0 / max(NdotV, 0.15);
                float atten = exp(-absorption * (thickness - 1.0));

                vec3 diffuse = water_color * NdotL * atten;
                vec3 specular = vec3(1.0) * spec;

                vec3 color = mix(diffuse, diffuse + specular, fresnel);

                float alpha = clamp(alpha_base + fresnel * 0.6, 0.02, 0.95);

                frag_color = vec4(color, alpha);
            }
        '''

        try:
            self.prog = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            print("✓ OpenGL shaders compiled successfully")
        except Exception as e:
            print(f"✗ Shader compilation failed: {e}")
            return False

        self._init_ssfr_resources()
        return True

    def _init_ssfr_resources(self):
        """初始化SSFR资源"""
        # Quad vertex shader
        quad_vs = r'''
            #version 330
            in vec2 in_pos;
            out vec2 v_uv;
            void main() {
                v_uv = in_pos * 0.5 + 0.5;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
        '''

        # Particle vertex shader
        particle_vs = r'''
        #version 330
        uniform mat4 projection;
        uniform mat4 view;
        uniform float u_particle_radius_world;
        uniform float u_viewport_h;
        uniform float u_point_scale;
        layout(location=0) in vec3 in_pos;
        out vec3 v_center_vs;
        void main() {
            vec4 view_pos = view * vec4(in_pos, 1.0);
            v_center_vs = view_pos.xyz;
            gl_Position = projection * view_pos;
            float z = max(-view_pos.z, 1e-4);
            float proj_y = projection[1][1];
            float r_px = (proj_y * u_particle_radius_world / z) * (u_viewport_h * 1.2);
            gl_PointSize = clamp(1.5 * r_px * u_point_scale, 1.0, 512.0);
        }
        '''

        # Depth fragment shader
        particle_depth_fs = r'''
        #version 330
        uniform mat4 projection;
        uniform float u_particle_radius_world;
        in vec3 v_center_vs;
        out float out_depth;
        void main() {
            vec2 p = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(p, p);
            if (r2 > 1.0) discard;
            float zcap = sqrt(1.0 - r2);
            float surf_z = v_center_vs.z + u_particle_radius_world * zcap;
            out_depth = max(-surf_z, 0.0);
            vec4 clip = projection * vec4(v_center_vs.xy, surf_z, 1.0);
            gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
        }
        '''

        # Thickness fragment shader
        particle_thick_fs = r'''
        #version 330
        uniform float u_particle_radius_world;
        out float out_thick;
        void main() {
            vec2 p = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(p, p);
            if (r2 > 1.0) discard;
            out_thick = 2.0 * u_particle_radius_world * sqrt(1.0 - r2);
        }
        '''

        # Bilateral filter shader
        bilateral_fs = r'''
        #version 330
        uniform sampler2D u_depth;
        uniform vec2 u_texel;
        uniform int u_filter_interval;
        uniform float u_sigma_xy;
        uniform float u_sigma_z;
        uniform int u_kernel_radius;
        in vec2 v_uv;
        out float frag_out;
        void main() {
            float center = texture(u_depth, v_uv).r;
            if (center < 1e-4) { frag_out = 0.0; return; }

            float sum = 0.0;
            float wsum = 0.0;

            for (int dy = -u_kernel_radius; dy <= u_kernel_radius; dy++) {
                for (int dx = -u_kernel_radius; dx <= u_kernel_radius; dx++) {
                    vec2 offset = vec2(float(dx), float(dy)) * float(u_filter_interval);
                    vec2 uv = v_uv + offset * u_texel;

                    float d = texture(u_depth, uv).r;
                    if (d < 1e-4) continue;

                    float d_spatial = length(offset);
                    float w_spatial = exp(-(d_spatial * d_spatial) / (2.0 * u_sigma_xy * u_sigma_xy));

                    float d_depth = abs(d - center);
                    float w_depth = exp(-(d_depth * d_depth) / (2.0 * u_sigma_z * u_sigma_z));

                    float w = w_spatial * w_depth;
                    sum += d * w;
                    wsum += w;
                }
            }

            frag_out = (wsum > 1e-6) ? (sum / wsum) : center;
        }
        '''

        # Shadow shaders
        shadow_vs = r'''
        #version 330
        uniform mat4 u_light_proj;
        uniform mat4 u_light_view;
        uniform float u_particle_radius_world;
        uniform float u_viewport_h;
        layout(location=0) in vec3 in_pos;
        out vec3 v_center_vs_light;
        void main() {
            vec4 view_pos = u_light_view * vec4(in_pos, 1.0);
            v_center_vs_light = view_pos.xyz;
            gl_Position = u_light_proj * view_pos;
            float z = max(-view_pos.z, 1e-3);
            float proj_y = u_light_proj[1][1];
            float point_size = (proj_y * u_particle_radius_world / z) * (u_viewport_h * 0.3) * 1.2;
            gl_PointSize = clamp(point_size, 1.0, 256.0);
        }
        '''

        shadow_fs = r'''
        #version 330
        uniform mat4 u_light_proj;
        uniform float u_particle_radius_world;
        in vec3 v_center_vs_light;
        out float out_depth;
        void main() {
            vec2 p = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(p, p);
            if (r2 > 1.0) discard;
            float zcap = sqrt(1.0 - r2);
            float surf_z = v_center_vs_light.z + u_particle_radius_world * zcap;
            vec4 clip = u_light_proj * vec4(v_center_vs_light.xy, surf_z, 1.0);
            out_depth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
            gl_FragDepth = out_depth;
        }
        '''

        # Composite shader
        comp_fs = r'''
        #version 330
        uniform sampler2D u_depth;
        uniform sampler2D u_thick;
        uniform sampler2D u_shadow;
        uniform vec2 u_texel;
        uniform vec2 u_shadow_texel;
        uniform float u_proj_x;
        uniform float u_proj_y;
        uniform mat4 u_view_inv;
        uniform mat4 u_light_vp;
        uniform vec3 u_light_dir_ws;
        uniform vec3 u_water_color;
        uniform float u_alpha_base;
        uniform float u_fresnel_power;
        uniform float u_spec_power;
        uniform float u_sigma_a;
        uniform float u_sigma_s;
        uniform float u_shadow_strength;
        uniform float u_caustic_strength;
        uniform float u_caustic_power;

        in vec2 v_uv;
        out vec4 frag_color;

        vec3 vs_pos(vec2 uv, float lin_d) {
            vec2 ndc = uv * 2.0 - 1.0;
            return vec3(ndc.x * lin_d / u_proj_x, ndc.y * lin_d / u_proj_y, -lin_d);
        }

        vec3 vs_normal(vec2 uv, float dc) {
            vec3 pc = vs_pos(uv, dc);
            float dr = texture(u_depth, uv + vec2(u_texel.x * 2.0, 0.0)).r;
            float dl = texture(u_depth, uv - vec2(u_texel.x * 2.0, 0.0)).r;
            float du = texture(u_depth, uv + vec2(0.0, u_texel.y * 2.0)).r;
            float dd = texture(u_depth, uv - vec2(0.0, u_texel.y * 2.0)).r;

            vec3 pr = vs_pos(uv + vec2(u_texel.x * 2.0, 0.0), dr);
            vec3 pl = vs_pos(uv - vec2(u_texel.x * 2.0, 0.0), dl);
            vec3 pu = vs_pos(uv + vec2(0.0, u_texel.y * 2.0), du);
            vec3 pd = vs_pos(uv - vec2(0.0, u_texel.y * 2.0), dd);

            vec3 tangentU_right = pr - pc;
            vec3 tangentU_left = pc - pl;
            vec3 tangentV_up = pu - pc;
            vec3 tangentV_down = pc - pd;

            bool use_left = (dl > 1e-4) && (dr < 1e-4 || abs(tangentU_left.z) < abs(tangentU_right.z));
            bool use_down = (dd > 1e-4) && (du < 1e-4 || abs(tangentV_down.z) < abs(tangentV_up.z));

            vec3 dx, dy;

            if (use_left && dl > 1e-4) dx = normalize(tangentU_left);
            else if (dr > 1e-4) dx = normalize(tangentU_right);
            else return vec3(0.0, 0.0, 1.0);

            if (use_down && dd > 1e-4) dy = normalize(tangentV_down);
            else if (du > 1e-4) dy = normalize(tangentV_up);
            else return vec3(0.0, 0.0, 1.0);

            vec3 n = cross(dx, dy);
            float len = length(n);
            if (len < 1e-8) return vec3(0.0, 0.0, 1.0);
            return n / len;
        }

        vec3 sky(vec3 dir) {
            float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
            vec3 zenith = vec3(0.10, 0.30, 0.80);
            vec3 horizon = vec3(0.70, 0.85, 1.00);
            vec3 ground = vec3(0.08, 0.08, 0.10);
            vec3 col = (dir.y >= 0.0) ? mix(horizon, zenith, pow(t, 0.6)) : mix(horizon, ground, pow(1.0 - t, 1.5));
            vec3 sun_dir = normalize(u_light_dir_ws);
            float sun = pow(max(dot(dir, sun_dir), 0.0), 256.0);
            col += vec3(1.0, 0.95, 0.80) * sun * 2.0;
            return col;
        }

        float sample_shadow(vec3 pos_ws) {
            vec4 lp = u_light_vp * vec4(pos_ws, 1.0);
            vec3 ndc = lp.xyz / lp.w;
            vec2 suv = ndc.xy * 0.5 + 0.5;
            if (suv.x < 0.01 || suv.x > 0.99 || suv.y < 0.01 || suv.y > 0.99)
                return 1.0;
            float cur = ndc.z * 0.5 + 0.5;
            float shadow = 0.0;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    float s = texture(u_shadow, suv + vec2(dx, dy) * u_shadow_texel).r;
                    shadow += (s + 0.003 < cur) ? (1.0 - u_shadow_strength) : 1.0;
                }
            }
            return shadow / 9.0;
        }

        void main() {
            float lin_d = texture(u_depth, v_uv).r;
            if (lin_d < 1e-4) discard;

            vec3 pos_vs = vs_pos(v_uv, lin_d);
            vec3 N_vs = vs_normal(v_uv, lin_d);

            vec3 pos_ws = (u_view_inv * vec4(pos_vs, 1.0)).xyz;
            vec3 N_ws = normalize(mat3(u_view_inv) * N_vs);
            vec3 cam_ws = (u_view_inv * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
            vec3 V = normalize(cam_ws - pos_ws);

            float NdotV = max(dot(N_ws, V), 0.0);

            float F0 = 0.02;
            float fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, u_fresnel_power);
            fresnel = min(fresnel, 0.85);

            float thick = clamp(texture(u_thick, v_uv).r, 0.0, 10.0);
            float trans = exp(-u_sigma_a * thick);
            float scatter = (1.0 - trans) * u_sigma_s;

            vec3 L = normalize(u_light_dir_ws);
            vec3 H = normalize(L + V);
            float NdotL = max(dot(N_ws, L), 0.0);
            float spec = pow(max(dot(N_ws, H), 0.0), u_spec_power);

            float shad = sample_shadow(pos_ws);

            vec3 refl_dir = reflect(-V, N_ws);
            vec3 refr_dir = refract(-V, N_ws, 1.0 / 1.333);
            vec3 env_refl = sky(refl_dir);
            vec3 env_refr = (length(refr_dir) > 0.5) ? sky(refr_dir) : sky(-V);

            // Beer-Lambert: transmitted environment attenuated by absorption (line below),
            // plus in-scattered fluid body color from direct lighting (line after)
            vec3 refr_color = trans * env_refr + (1.0 - trans) * u_water_color;
            refr_color += u_water_color * scatter * NdotL * shad;

            vec3 color = mix(refr_color, env_refl, fresnel);
            color += vec3(1.0) * spec * (0.3 + 0.7 * fresnel) * shad;

            float tC = texture(u_thick, v_uv).r;
            float tR = texture(u_thick, v_uv + vec2(u_texel.x, 0.0)).r;
            float tU = texture(u_thick, v_uv + vec2(0.0, u_texel.y)).r;
            float cgrad = length(vec2(tR - tC, tU - tC));
            float caustic_factor = clamp(cgrad * 4.0, 0.0, 1.0);
            float caustic = pow(caustic_factor, u_caustic_power) * u_caustic_strength;
            caustic = min(caustic, 0.4);
            color += caustic * vec3(1.0, 0.96, 0.82) * shad;

            float alpha = clamp(u_alpha_base + (1.0 - trans) * 0.85 + fresnel * 0.25, 0.05, 0.98);

            frag_color = vec4(color, alpha);
        }
        '''

        # Compile programs
        self.particle_depth_prog = self.ctx.program(vertex_shader=particle_vs, fragment_shader=particle_depth_fs)
        self.particle_thick_prog = self.ctx.program(vertex_shader=particle_vs, fragment_shader=particle_thick_fs)
        self.bilateral_prog = self.ctx.program(vertex_shader=quad_vs, fragment_shader=bilateral_fs)
        self.composite_prog = self.ctx.program(vertex_shader=quad_vs, fragment_shader=comp_fs)
        self.shadow_prog = self.ctx.program(vertex_shader=shadow_vs, fragment_shader=shadow_fs)

        # Fullscreen quad
        quad = np.array([
            -1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
            -1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
        ], dtype=np.float32)
        self.quad_vbo = self.ctx.buffer(quad.tobytes())

        self.quad_vao_bilateral = self.ctx.vertex_array(self.bilateral_prog, [(self.quad_vbo, '2f', 'in_pos')])
        self.quad_vao_comp = self.ctx.vertex_array(self.composite_prog, [(self.quad_vbo, '2f', 'in_pos')])

        # Particle VBO / VAOs
        self.particle_vbo = self.ctx.buffer(reserve=CONFIG.MAX_PARTICLES * 3 * 4)
        self.particle_vao_depth = self.ctx.vertex_array(self.particle_depth_prog, [(self.particle_vbo, '3f', 'in_pos')])
        self.particle_vao_thick = self.ctx.vertex_array(self.particle_thick_prog, [(self.particle_vbo, '3f', 'in_pos')])
        self.shadow_vao = self.ctx.vertex_array(self.shadow_prog, [(self.particle_vbo, '3f', 'in_pos')])

    def update_mesh_data(self, vertices, triangles):
        """Update mesh vertex and index data - GPU优化版"""
        if vertices is None or triangles is None:
            return

        if len(vertices) == 0 or len(triangles) == 0:
            self.vertices = []
            self.triangles = []
            self.vertex_count = 0
            self.index_count = 0
            return

        self.vertices = vertices
        self.triangles = triangles

        t0 = time.perf_counter()
        vertex_data = self._create_vertex_data(vertices, triangles)
        indices_u32 = np.asarray(triangles, dtype=np.uint32).reshape(-1)
        t_cpu = time.perf_counter() - t0

        vertex_bytes = vertex_data.nbytes
        index_bytes = indices_u32.nbytes

        recreate_buffers = (
            self.vertex_buffer is None or
            self.index_buffer is None or
            vertex_bytes > self.vertex_buffer.size or
            index_bytes > self.index_buffer.size
        )

        if recreate_buffers:
            if self.vertex_buffer:
                self.vertex_buffer.release()
            if self.index_buffer:
                self.index_buffer.release()
            if self.vao:
                self.vao.release()

            self.vertex_buffer = self.ctx.buffer(reserve=max(vertex_bytes, self.max_vertices * 24))
            self.index_buffer = self.ctx.buffer(reserve=max(index_bytes, self.max_indices * 4))

            self.vao = self.ctx.vertex_array(
                self.prog,
                [(self.vertex_buffer, '3f 3f', 0, 1)],
                self.index_buffer,
                index_element_size=4,
            )

        t1 = time.perf_counter()
        self.vertex_buffer.write(vertex_data.tobytes())
        self.index_buffer.write(indices_u32.tobytes())
        t_gpu = time.perf_counter() - t1

        self.vertex_count = int(vertex_data.shape[0])
        self.index_count = int(indices_u32.size)

        self._tlog.print(
            f"upload: cpu_prep={t_cpu*1000:.1f}ms gpu_write={t_gpu*1000:.1f}ms "
            f"verts={self.vertex_count} idx={self.index_count}"
        )

        self.mesh_system.performance_stats['gpu_upload_time'] = t_gpu

    def _ensure_ssfr_targets(self, width: int, height: int):
        w = max(1, int(width * self.ssfr_scale))
        h = max(1, int(height * self.ssfr_scale))
        if w == self._ssfr_w and h == self._ssfr_h and getattr(self, "depth_fbo0", None) is not None:
            return

        self._ssfr_w, self._ssfr_h = w, h

        # Release old resources
        for obj in (
            getattr(self, "depth_fbo0", None),
            getattr(self, "depth_fbo1", None),
            getattr(self, "thick_fbo", None),
            getattr(self, "depth_tex0", None),
            getattr(self, "depth_tex1", None),
            getattr(self, "thick_tex", None),
            getattr(self, "_depth_rb0", None),
            getattr(self, "_depth_rb1", None),
        ):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass

        # Linear depth textures
        self.depth_tex0 = self.ctx.texture((w, h), components=1, dtype='f4')
        self.depth_tex1 = self.ctx.texture((w, h), components=1, dtype='f4')
        for t in (self.depth_tex0, self.depth_tex1):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Depth renderbuffers
        self._depth_rb0 = self.ctx.depth_renderbuffer((w, h))
        self._depth_rb1 = self.ctx.depth_renderbuffer((w, h))

        # Thickness texture
        self.thick_tex = self.ctx.texture((w, h), components=1, dtype='f2')
        self.thick_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # FBOs
        self.depth_fbo0 = self.ctx.framebuffer(
            color_attachments=[self.depth_tex0], depth_attachment=self._depth_rb0)
        self.depth_fbo1 = self.ctx.framebuffer(
            color_attachments=[self.depth_tex1], depth_attachment=self._depth_rb1)
        self.thick_fbo = self.ctx.framebuffer(color_attachments=[self.thick_tex])

    def _ensure_shadow_targets(self, width: int, height: int):
        w = CONFIG.SHADOW_MAP_SIZE
        h = CONFIG.SHADOW_MAP_SIZE
        if w == self.shadow_w and h == self.shadow_h and self.shadow_fbo is not None:
            return

        self.shadow_w, self.shadow_h = w, h

        # Release old shadow resources
        for obj in (self.shadow_fbo, self.shadow_tex, getattr(self, "_shadow_depth_rb", None)):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass

        self.shadow_tex = self.ctx.texture((w, h), components=1, dtype='f4')
        self.shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.shadow_tex.repeat_x = False
        self.shadow_tex.repeat_y = False

        self._shadow_depth_rb = self.ctx.depth_renderbuffer((w, h))
        self.shadow_fbo = self.ctx.framebuffer(
            color_attachments=[self.shadow_tex], depth_attachment=self._shadow_depth_rb)

    def update_particles(self, positions: np.ndarray):
        """positions: shape (N,3) float32"""
        if positions is None:
            self.particle_count = 0
            return
        n = int(len(positions))
        self.particle_count = n
        if n == 0 or self.particle_vbo is None:
            return

        pos = np.asarray(positions, dtype=np.float32)
        if not pos.flags['C_CONTIGUOUS']:
            pos = np.ascontiguousarray(pos)

        byte_len = n * 3 * 4
        if byte_len > self.particle_vbo.size:
            self.particle_vbo.release()
            self.particle_vbo = self.ctx.buffer(reserve=max(byte_len, CONFIG.MAX_PARTICLES * 3 * 4))

            if self.particle_vao_depth:
                self.particle_vao_depth.release()
            if self.particle_vao_thick:
                self.particle_vao_thick.release()
            if self.shadow_vao:
                self.shadow_vao.release()

            self.particle_vao_depth = self.ctx.vertex_array(
                self.particle_depth_prog, [(self.particle_vbo, '3f', 'in_pos')]
            )
            self.particle_vao_thick = self.ctx.vertex_array(
                self.particle_thick_prog, [(self.particle_vbo, '3f', 'in_pos')]
            )
            self.shadow_vao = self.ctx.vertex_array(
                self.shadow_prog, [(self.particle_vbo, '3f', 'in_pos')]
            )

        self.particle_vbo.write(pos[:n].tobytes())

    def _create_vertex_data(self, vertices, triangles):
        """
        更快/更正确的法线计算版本
        """
        verts = np.asarray(vertices, dtype=np.float32)
        tris = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
        vert_count = verts.shape[0]
        tri_count = tris.shape[0]

        if vert_count == 0 or tri_count == 0:
            if self._vertex_data is None or self._vertex_data.shape[0] < vert_count:
                self._vertex_data = np.empty((max(vert_count, 1), 6), dtype=np.float32)
            vd = self._vertex_data[:vert_count]
            if vert_count:
                vd[:, :3] = verts
                vd[:, 3:] = 0
            return vd

        if self._temp_normals is None or self._temp_normals.shape[0] < vert_count:
            self._temp_normals = np.zeros((vert_count, 3), dtype=np.float32)
            self._temp_counts = np.zeros((vert_count,), dtype=np.int32)
        normals = self._temp_normals[:vert_count]
        counts = self._temp_counts[:vert_count]
        normals.fill(0.0)
        counts.fill(0)

        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        lens = np.linalg.norm(face_normals, axis=1)
        mask = lens > 1e-12
        face_normals[mask] /= lens[mask][:, None]

        np.add.at(normals, tris[:, 0], face_normals)
        np.add.at(normals, tris[:, 1], face_normals)
        np.add.at(normals, tris[:, 2], face_normals)
        np.add.at(counts, tris[:, 0], 1)
        np.add.at(counts, tris[:, 1], 1)
        np.add.at(counts, tris[:, 2], 1)

        m = counts > 0
        normals[m] /= counts[m][:, None]
        nlen = np.linalg.norm(normals, axis=1)
        m2 = nlen > 1e-12
        normals[m2] /= nlen[m2][:, None]

        if self._vertex_data is None or self._vertex_data.shape[0] < vert_count:
            self._vertex_data = np.empty((vert_count, 6), dtype=np.float32)
        vertex_data = self._vertex_data[:vert_count]
        vertex_data[:, :3] = verts
        vertex_data[:, 3:] = normals
        return vertex_data

    def render(self, width, height, screen_fbo=None):
        """Render the scene"""
        if not self.ctx:
            return

        if self.use_ssfr:
            self._render_ssfr(width, height, screen_fbo)
            return

        # Original mesh rendering logic
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.clear(0.1, 0.1, 0.15, 1.0)

        fbo = self.ctx.detect_framebuffer()
        fbo.use()

        if self.vao is None or self.index_count == 0:
            return

        projection = create_perspective_matrix(60.0, width / height, 0.1, 100.0)
        cam_pos = calculate_camera_position(self.camera_distance, self.camera_angle_x, self.camera_angle_y, self.camera_target)
        view = create_lookat_matrix(cam_pos, self.camera_target, [0.0, 1.0, 0.0])
        model = np.eye(4, dtype=np.float32)

        self.prog['projection'].write(projection.astype('f4').tobytes())
        self.prog['view'].write(view.astype('f4').tobytes())
        self.prog['model'].write(model.astype('f4').tobytes())
        self.prog['light_pos'].value = (5.0, 8.0, 5.0)
        self.prog['camera_pos'].value = tuple(cam_pos)

        # Water material parameters
        self.prog['water_color'].value = CONFIG.WATER_COLOR
        self.prog['alpha_base'].value = CONFIG.ALPHA_BASE
        self.prog['fresnel_power'].value = CONFIG.FRESNEL_POWER
        self.prog['spec_power'].value = CONFIG.SPEC_POWER
        self.prog['absorption'].value = 2.0

        self.vao.render(moderngl.TRIANGLES, vertices=self.index_count)

    def _render_ssfr(self, width: int, height: int, screen_fbo):
        self._ensure_ssfr_targets(width, height)
        self._ensure_shadow_targets(width, height)

        near, far = 0.1, 100.0
        projection = create_perspective_matrix(60.0, width / height, near, far)
        cam_pos = calculate_camera_position(self.camera_distance, self.camera_angle_x, self.camera_angle_y, self.camera_target)
        view = create_lookat_matrix(cam_pos, self.camera_target, [0.0, 1.0, 0.0])
        view_inv = np.linalg.inv(view).astype(np.float32)

        r_world = float(self.ssfr_particle_radius_world)

        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # Pass 1 - Depth
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        self.depth_fbo0.use()
        self.ctx.viewport = (0, 0, self._ssfr_w, self._ssfr_h)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        if self.particle_count > 0:
            p = self.particle_depth_prog
            p['projection'].write(projection.tobytes())
            p['view'].write(view.tobytes())
            p['u_particle_radius_world'].value = r_world
            p['u_viewport_h'].value = float(self._ssfr_h)
            p['u_point_scale'].value = 1.0
            self.particle_vao_depth.render(mode=moderngl.POINTS, vertices=self.particle_count)

        # Pass 2 - Bilateral filtering with clean ping-pong logic
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # Perform bilateral filtering with proper ping-pong
        final_depth_tex = self._filter_depth_passes()

        # Save depth texture if needed
        current_time = time.time()
        if current_time - self.last_depth_save_time >= self.depth_save_interval:
            self._save_depth_texture("depth_tex0")
            self.last_depth_save_time = current_time
            self.depth_frame_counter += 1

        # Pass 3 - Thickness
        self.thick_fbo.use()
        self.ctx.viewport = (0, 0, self._ssfr_w, self._ssfr_h)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)

        if self.particle_count > 0:
            p = self.particle_thick_prog
            p['projection'].write(projection.tobytes())
            p['view'].write(view.tobytes())
            p['u_particle_radius_world'].value = r_world
            p['u_viewport_h'].value = float(self._ssfr_h)
            p['u_point_scale'].value = 1.0
            self.particle_vao_thick.render(mode=moderngl.POINTS, vertices=self.particle_count)

        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # Pass 4 - Shadow
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)

        light_proj = create_perspective_matrix(70.0, 1.0, 0.5, 60.0)
        light_view = create_lookat_matrix(self.light_pos.tolist(), self.light_target.tolist(), [0.0, 1.0, 0.0])
        light_vp = (light_proj @ light_view).astype(np.float32)
        light_dir_ws = (self.light_target - self.light_pos)
        light_dir_ws = light_dir_ws / (np.linalg.norm(light_dir_ws) + 1e-8)

        self.shadow_fbo.use()
        self.ctx.viewport = (0, 0, self.shadow_w, self.shadow_h)
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)

        if self.particle_count > 0:
            p = self.shadow_prog
            p['u_light_proj'].write(light_proj.tobytes())
            p['u_light_view'].write(light_view.tobytes())
            p['u_particle_radius_world'].value = r_world
            p['u_viewport_h'].value = float(self.shadow_h)
            self.shadow_vao.render(mode=moderngl.POINTS, vertices=self.particle_count)

        # Pass 5 - Composite
        screen_fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.clear(0.06, 0.08, 0.12, 1.0)

        cp = self.composite_prog

        cp['u_texel'].value = (1.0 / self._ssfr_w, 1.0 / self._ssfr_h)
        cp['u_proj_x'].value = float(projection[0, 0])
        cp['u_proj_y'].value = float(projection[1, 1])
        cp['u_view_inv'].write(view_inv.tobytes())

        cp['u_light_vp'].write(light_vp.tobytes())
        cp['u_light_dir_ws'].value = tuple(light_dir_ws.astype(np.float32).tolist())
        cp['u_shadow_texel'].value = (1.0 / self.shadow_w, 1.0 / self.shadow_h)

        cp['u_water_color'].value = CONFIG.WATER_COLOR
        cp['u_alpha_base'].value = CONFIG.ALPHA_BASE
        cp['u_fresnel_power'].value = CONFIG.FRESNEL_POWER
        cp['u_spec_power'].value = CONFIG.SPEC_POWER
        cp['u_sigma_a'].value = CONFIG.SIGMA_A
        cp['u_sigma_s'].value = CONFIG.SIGMA_S

        cp['u_shadow_strength'].value = CONFIG.SHADOW_STRENGTH
        cp['u_caustic_strength'].value = CONFIG.CAUSTIC_STRENGTH
        cp['u_caustic_power'].value = CONFIG.CAUSTIC_POWER

        final_depth_tex.use(location=0)
        self.thick_tex.use(location=1)
        self.shadow_tex.use(location=2)
        cp['u_depth'].value = 0
        cp['u_thick'].value = 1
        cp['u_shadow'].value = 2

        self.quad_vao_comp.render(moderngl.TRIANGLES, vertices=6)

    def render_default_scene(self, width, height):
        """渲染默认场景"""
        if not self.ctx or not self.prog:
            return

        try:
            self.ctx.clear(0.1, 0.1, 0.15, 1.0)
            fbo = self.ctx.detect_framebuffer()
            fbo.use()

            if not hasattr(self, '_cached_projection'):
                self._cached_projection = np.empty((4, 4), dtype=np.float32)
                self._cached_view = np.empty((4, 4), dtype=np.float32)
                self._cached_model = np.eye(4, dtype=np.float32)

            create_perspective_matrix(60.0, width / height, 0.1, 100.0, out=self._cached_projection)
            camera_pos = calculate_camera_position(self.camera_distance, self.camera_angle_x, self.camera_angle_y, self.camera_target)
            create_lookat_matrix(camera_pos, self.camera_target, [0.0, 1.0, 0.0], out=self._cached_view)

            self.prog['projection'].write(self._cached_projection.astype('f4').tobytes())
            self.prog['view'].write(self._cached_view.astype('f4').tobytes())
            self.prog['model'].write(self._cached_model.astype('f4').tobytes())
            self.prog['light_pos'].value = (5.0, 5.0, 5.0)
            self.prog['camera_pos'].value = tuple(camera_pos)

        except Exception as e:
            print(f"Default scene rendering error: {e}")

    def cleanup(self):
        """Clean up OpenGL resources"""
        resources = [
            'vao', 'vertex_buffer', 'index_buffer', 'prog',
            'depth_fbo0', 'depth_fbo1', 'thick_fbo',
            'depth_tex0', 'depth_tex1', 'thick_tex',
            'particle_vao_depth', 'particle_vao_thick',
            'particle_depth_prog', 'particle_thick_prog',
            'bilateral_prog', 'quad_vao_bilateral', 'quad_vao_comp'
        ]

        for attr_name in resources:
            obj = getattr(self, attr_name, None)
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass

    def _filter_depth_passes(self):
        """执行双边滤波，返回最终的深度纹理"""
        bp = self.bilateral_prog
        bp['u_texel'].value = (1.0 / self._ssfr_w, 1.0 / self._ssfr_h)
        bp['u_sigma_xy'].value = float(self.ssfr_blur_sigma)
        bp['u_sigma_z'].value = float(self.ssfr_depth_sigma)
        
        # 初始化乒乓缓冲
        src_tex = self.depth_tex0
        dst_fbo = self.depth_fbo1

        # 执行5次滤波，间隔按指数增长: 2, 4, 8, 16, 32
        bp['u_kernel_radius'].value = 2
        for i in range(1, 6):
            interval = 2 ** i
            dst_fbo.use()
            self.ctx.viewport = (0, 0, self._ssfr_w, self._ssfr_h)
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            bp['u_filter_interval'].value = interval
            src_tex.use(location=0)
            bp['u_depth'].value = 0
            self.quad_vao_bilateral.render(moderngl.TRIANGLES, vertices=6)
            # 乒乓交换
            src_tex, dst_fbo = (
                self.depth_tex1 if src_tex == self.depth_tex0 else self.depth_tex0,
                self.depth_fbo0 if dst_fbo == self.depth_fbo1 else self.depth_fbo1
            )

        # After 5 iterations (odd count): src_tex=depth_tex1, dst_fbo=depth_fbo0
        # 9x9核精炼 (interval=1): reads depth_tex1 → writes depth_fbo0 (depth_tex0), no hazard
        bp['u_kernel_radius'].value = 4
        bp['u_filter_interval'].value = 1
        dst_fbo.use()
        self.ctx.viewport = (0, 0, self._ssfr_w, self._ssfr_h)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        src_tex.use(location=0)
        bp['u_depth'].value = 0
        self.quad_vao_bilateral.render(moderngl.TRIANGLES, vertices=6)

        # Result is in the texture attached to dst_fbo
        # (depth_fbo0 attaches depth_tex0; depth_fbo1 attaches depth_tex1)
        final_tex = self.depth_tex0 if dst_fbo == self.depth_fbo0 else self.depth_tex1
        return final_tex

    def _save_depth_texture(self, name):
        """保存深度纹理到文件"""
        if not self.depth_save_enabled or not hasattr(self, 'depth_tex0') or self.depth_tex0 is None:
            return
        
        try:
            import os
            import datetime
            
            # 从GPU读取深度纹理数据
            depth_data = self._read_depth_texture(name)
            if depth_data is None:
                return
            
            # 生成文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            frame_num = self.depth_frame_counter
            filename = f"depth_frame_{frame_num:06d}_{timestamp}.{self.depth_save_format}"
            filepath = os.path.join(self.depth_save_path, filename)
            
            # 保存数据
            if self.depth_save_format.lower() == 'png':
                self._save_depth_as_png(depth_data, filepath)
            elif self.depth_save_format.lower() == 'npy':
                self._save_depth_as_npy(depth_data, filepath)
            else:
                # 默认保存为PNG
                self._save_depth_as_png(depth_data, filepath)
            
            print(f"✓ 深度纹理已保存: {filepath}")
            
        except Exception as e:
            print(f"✗ 深度纹理保存失败: {e}")
    
    def _read_depth_texture(self, name):
        """从GPU读取深度纹理数据"""
        try:
            # 确保有有效的深度纹理
            try:
                target = getattr(self, name, None)
            except:
                target = name
            if target is None:
                print("✗ 深度纹理未初始化")
                return None
            
            # 获取纹理尺寸
            width, height = target.size
            
            # 创建临时FBO来读取纹理数据
            temp_fbo = self.ctx.framebuffer(color_attachments=[target])
            
            # 读取浮点数据
            depth_bytes = temp_fbo.read(
                viewport=(0, 0, width, height),
                components=1,
                dtype='f4'
            )
            
            # 释放临时FBO
            temp_fbo.release()
            
            # 转换为numpy数组
            import numpy as np
            depth_data = np.frombuffer(depth_bytes, dtype=np.float32)
            depth_data = depth_data.reshape((height, width))
            
            return depth_data
            
        except Exception as e:
            print(f"✗ 读取深度纹理失败: {e}")
            return None
    
    def _save_depth_as_png(self, depth_data, filepath):
        """将深度数据保存为PNG格式"""
        try:
            import numpy as np
            from PIL import Image
            
            # 处理深度数据
            if self.depth_normalize_range:
                # 归一化到[0,1]范围
                depth_min = np.min(depth_data[depth_data > 0]) if np.any(depth_data > 0) else 0
                depth_max = np.max(depth_data)
                if depth_max > depth_min:
                    normalized = (depth_data - depth_min) / (depth_max - depth_min)
                else:
                    normalized = depth_data
                # 转换为8位灰度图像
                image_data = (normalized * 255).astype(np.uint8)
            else:
                # 直接转换，可能丢失精度
                image_data = (depth_data * 255).astype(np.uint8)
            
            # 创建并保存图像
            image = Image.fromarray(image_data, mode='L')
            image.save(filepath, format='PNG', compress_level=1)
            
        except Exception as e:
            print(f"✗ 保存PNG失败: {e}")
            raise
    
    def _save_depth_as_npy(self, depth_data, filepath):
        """将深度数据保存为NumPy格式"""
        try:
            import numpy as np
            np.save(filepath, depth_data)
        except Exception as e:
            print(f"✗ 保存NPY失败: {e}")
            raise