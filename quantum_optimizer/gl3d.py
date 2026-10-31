"""Cinematic GPU 3D renderer for the Legionary viewer (moderngl + GLSL 3.30 core).

Tuned for older Mac GPUs (Iris Plus 640/650, Radeon Pro 555/560; OpenGL 4.1, Retina):
 * the 3D scene renders at an internal scale (default 0.75 of the window, 0.5 on Retina
   framebuffers) and is upscaled with bilinear filtering in the composite;
 * the star field / cosmic web is baked ONCE into an equirectangular texture (per theme); the
   stratosphere is baked once into a small lon/lat texture (spin = u offset);
 * fog is exp2, evaluated per vertex; the flame ring is analytic polar math (no noise);
 * bloom = one bright-pass that downsamples straight to 1/4 res + one H + one V blur that use
   bilinear tap merging (3 fetches = 5-tap gaussian);
 * instances are rebuilt only when the circuit changes or something animates, frustum-culled
   into preallocated numpy buffers, flames only for near live cubes (LOD; idle ones at pilot-light strength), and drawn with
   one instanced call each: spheres, edges (solid/dashed/dotted via an instance attribute),
   ribbons, flames. 10 draw calls per frame in total;
 * the pygame UI overlay is re-uploaded only when its content changes.
Window: pygame OPENGL (3.3 core, forward-compatible for macOS), vsync + 60 fps cap. Screenshots:
moderngl standalone context (EGL / llvmpipe on a headless box)."""
import math, os, time

import numpy as np

from . import scene3d, viewer

COMMON = """
uniform mat4 u_vp; uniform vec3 u_eye; uniform float u_fog; uniform vec3 u_bg; uniform float u_time;
float fogk(vec3 p){ return exp2(-u_fog * 1.442695 * length(p - u_eye)); }
"""

BG_VS = """#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos; gl_Position = vec4(in_pos, 0.9999, 1.0); }"""
# bake: one pass into an equirectangular texture (v_uv in [-1,1]^2 -> lon/lat -> direction)
BG_BAKE_FS = """#version 330
in vec2 v_uv; out vec4 o;
uniform vec3 u_bg; uniform vec3 u_web; uniform float u_density;
float h(vec3 p){ p = fract(p * 0.3183099 + .1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float n3(vec3 x){ vec3 i = floor(x), f = fract(x); f = f*f*(3.-2.*f);
  return mix(mix(mix(h(i),h(i+vec3(1,0,0)),f.x), mix(h(i+vec3(0,1,0)),h(i+vec3(1,1,0)),f.x),f.y),
             mix(mix(h(i+vec3(0,0,1)),h(i+vec3(1,0,1)),f.x), mix(h(i+vec3(0,1,1)),h(i+vec3(1,1,1)),f.x),f.y), f.z); }
void main(){
  float lon = v_uv.x * 3.14159265, lat = v_uv.y * 1.5707963;
  vec3 d = vec3(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
  vec3 c = u_bg * (0.7 + 0.5 * (1.0 - abs(d.y)));
  float web = 0.0, amp = 0.5; vec3 q = d * 3.0;
  for (int i = 0; i < 4; i++){ web += amp * n3(q); q *= 2.1; amp *= 0.5; }
  c += u_web * smoothstep(0.55, 0.95, web) * 0.10;
  vec3 g = floor(d * 260.0); float s = h(g); float thr = 1.0 - 0.006 * u_density;
  if (s > thr) { vec3 f = fract(d * 260.0) - 0.5;
    c += vec3(0.85, 0.85, 1.0) * smoothstep(0.3, 0.0, length(f)) * (s - thr) / (1.0 - thr) * 2.6; }
  o = vec4(c, 1.0);
}"""
BG_FS = """#version 330
in vec2 v_uv; out vec4 o; uniform mat4 u_inv; uniform sampler2D u_sky;
void main(){
  vec4 a = u_inv * vec4(v_uv, 1.0, 1.0); vec3 d = normalize(a.xyz / a.w);
  vec2 uv = vec2(atan(d.x, d.z) / 6.2831853 + 0.5, asin(clamp(d.y, -1.0, 1.0)) / 3.14159265 + 0.5);
  o = vec4(textureLod(u_sky, uv, 0.0).rgb, 1.0);   // explicit LOD: no seam artefact at the wrap
}"""

GRID_VS = """#version 330
""" + COMMON + """
in vec2 in_pos; out vec3 v_w;
uniform float u_y; uniform vec2 u_c;
void main(){ v_w = vec3(u_c.x + in_pos.x * 400.0, u_y, u_c.y + in_pos.y * 400.0); gl_Position = u_vp * vec4(v_w, 1.0); }"""
GRID_FS = """#version 330
""" + COMMON + """
in vec3 v_w; out vec4 o; uniform vec3 u_col;
void main(){
  vec2 g = abs(fract(v_w.xz / vec2(3.4, 2.7) - 0.5) - 0.5) / fwidth(v_w.xz / vec2(3.4, 2.7));
  float line = 1.0 - min(min(g.x, g.y), 1.0);
  float k = fogk(v_w) * fogk(v_w);
  o = vec4(u_col * line * 0.16 * k, 1.0);
}"""

SPHERE_VS = """#version 330
""" + COMMON + """
in vec2 in_corner;
in vec3 i_pos; in float i_half; in float i_state; in float i_spin; in float i_alpha;
out vec2 v_uv; out float v_spin; out float v_alpha; out float v_fog; out float v_work;
uniform vec3 u_right; uniform vec3 u_up;
void main(){
  float r = i_half * 0.58;
  v_uv = in_corner; v_spin = i_spin; v_fog = fogk(i_pos);
  v_alpha = (i_state < 1.5) ? i_alpha : 0.0;   // removed / failed cubes are empty frames
  v_work = (abs(i_state - 1.0) < 0.1) ? 1.0 : 0.0;
  vec3 w = i_pos + (u_right * in_corner.x + u_up * in_corner.y) * r;
  gl_Position = u_vp * vec4(w, 1.0);
  if (v_alpha <= 0.0) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}"""
# stratosphere baked once to a lon/lat texture (u = lon/2pi, v = lat mapped from [-1,1])
SKYTEX_FS = """#version 330
in vec2 v_uv; out vec4 o;
uniform vec3 u_s0, u_s1, u_s2, u_s3; uniform float u_stars;
float h2(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main(){
  float lon = (v_uv.x * 0.5 + 0.5) * 6.2831853; float lat = v_uv.y;
  vec3 c = lat > 0.25 ? mix(u_s1, u_s0, smoothstep(0.25, 0.95, lat)) : mix(u_s2, u_s1, smoothstep(-0.05, 0.25, lat));
  c += u_s3 * exp(-pow((lat + 0.02) / 0.06, 2.0)) * 0.85;
  c = lat < -0.08 ? mix(c, u_s2 * 0.25, smoothstep(-0.08, -0.6, lat)) : c;
  vec2 g = floor(vec2(lon * 9.0, lat * 18.0)); float s = h2(g);
  if (lat > 0.05 && s > 1.0 - 0.08 * u_stars) { vec2 f = fract(vec2(lon * 9.0, lat * 18.0)) - 0.5;
    c += vec3(1.0, 0.92, 0.8) * smoothstep(0.25, 0.0, length(f)) * 1.5; }
  o = vec4(c, 1.0);
}"""
SPHERE_FS = """#version 330
in vec2 v_uv; in float v_spin; in float v_alpha; in float v_fog; in float v_work; out vec4 o;
uniform vec3 u_s3; uniform vec3 u_bg; uniform sampler2D u_strat;
void main(){
  float r2 = dot(v_uv, v_uv); if (r2 > 1.0) discard;
  float nz = sqrt(1.0 - r2);
  float lon = atan(v_uv.x, nz) + v_spin;
  vec3 c = textureLod(u_strat, vec2(lon * 0.1591549, v_uv.y * 0.5 + 0.5), 0.0).rgb;  // LOD 0: no wrap seam
  float rim = 1.0 - nz; rim *= rim;
  c += u_s3 * rim * rim * 0.25;                       // atmosphere rim
  c *= 0.75 + 0.3 * v_work;
  o = vec4(mix(u_bg, c, v_fog), v_alpha * 0.95);
}"""

EDGE_VS = """#version 330
""" + COMMON + """
in vec3 in_a; in vec3 in_b; in float in_along; in float in_side;
in vec3 i_pos; in float i_half; in float i_state; in float i_alpha; in vec3 i_col; in float i_style; in float i_work;
out float v_t; out float v_side; out vec3 v_col; out float v_a; out float v_style;
uniform float u_glow; uniform float u_lod_far;
void main(){
  vec3 A = i_pos + in_a * i_half, B = i_pos + in_b * i_half;
  vec3 P = mix(A, B, in_along);
  vec3 dir = normalize(B - A);
  vec3 side = normalize(cross(dir, P - u_eye));
  float i_alpha_k = 1.0;
  float wdt = i_half * (0.022 + 0.02 * i_work) * (i_state > 1.5 ? 0.8 : 1.0);
  if (length(i_pos - u_eye) > u_lod_far && i_style > 0.5) i_alpha_k = 0.5;
  P += side * in_side * wdt * 2.5;
  v_t = in_along * 2.0; v_side = in_side;
  v_style = (length(i_pos - u_eye) > u_lod_far) ? 0.0 : i_style;   // LOD: far frames solid
  float boost = (i_state > 1.5) ? 2.2 : u_glow * (1.0 + 0.8 * i_work);
  v_col = i_col * boost * fogk(P); v_a = i_alpha * i_alpha_k;
  gl_Position = u_vp * vec4(P, 1.0);
}"""
EDGE_FS = """#version 330
in float v_t; in float v_side; in vec3 v_col; in float v_a; in float v_style; out vec4 o;
void main(){
  if (v_style > 0.5 && v_style < 1.5 && fract(v_t * 3.0) > 0.55) discard;   // dashed
  float prof = 1.0 - v_side * v_side;
  if (v_style > 1.5) {                                                       // dotted
    float d = length(vec2((fract(v_t * 7.0) - 0.5) * 3.0, v_side));
    if (d > 1.0) discard; prof = 1.0 - d * d; }
  float core = smoothstep(0.55, 1.0, prof);
  o = vec4(v_col * (prof * 0.12 + core * 0.55) * v_a, 1.0);
}"""

RIBBON_VS = """#version 330
""" + COMMON + """
in vec2 in_ts;
in vec3 i_p0; in vec3 i_p1; in float i_width; in float i_act; in float i_alpha; in float i_state; in float i_arc; in float i_phase;
out float v_t; out float v_side; out float v_act; out float v_a; out float v_grey; out float v_fog; out float v_phase;
void main(){
  float t = in_ts.x;
  vec3 P = mix(i_p0, i_p1, t) + vec3(0.0, sin(3.14159 * t) * i_arc, 0.0);
  vec3 d = normalize(i_p1 - i_p0 + vec3(0.0, cos(3.14159 * t) * i_arc * 3.14159, 0.0));
  vec3 side = normalize(cross(d, P - u_eye));
  P += side * in_ts.y * i_width;
  v_t = t; v_side = in_ts.y; v_act = i_act; v_a = i_alpha; v_grey = i_state > 1.5 ? 1.0 : 0.0;
  v_fog = fogk(P); v_phase = i_phase;
  gl_Position = u_vp * vec4(P, 1.0);
}"""
RIBBON_FS = """#version 330
in float v_t; in float v_side; in float v_act; in float v_a; in float v_grey; in float v_fog; in float v_phase; out vec4 o;
uniform vec3 u_c0, u_c1, u_c2, u_grey; uniform float u_time; uniform float u_glow;
void main(){
  float prof = 1.0 - v_side * v_side;
  vec3 c = v_t < 0.5 ? mix(u_c0, u_c1, v_t * 2.0) : mix(u_c1, u_c2, v_t * 2.0 - 1.0);
  float flow = 0.55 + 0.45 * sin((v_t * 6.0 - u_time * (1.2 + 2.5 * v_act) + v_phase * 6.28) * 6.2832);
  vec3 col = c * (0.35 + 0.65 * flow) * (prof * (0.6 + u_glow) + smoothstep(0.6, 1.0, prof) * 1.2) * (0.6 + 1.2 * v_act);
  if (v_grey > 0.5) col = u_grey * prof * 0.5;
  o = vec4(col * v_a * v_fog, 1.0);
}"""

FLAME_VS = """#version 330
""" + COMMON + """
in vec2 in_corner;
in vec3 i_pos; in float i_half; in float i_state; in float i_spin; in float i_alpha; in float i_work;
out vec2 v_uv; out float v_k; out float v_spin; out float v_seed;
uniform vec3 u_right; uniform vec3 u_up;
void main(){
  bool on = i_state < 1.5;   // idle + working: idle keeps a dim pilot-light ring
  float r = i_half * 0.58;
  v_uv = in_corner * 1.85; v_spin = i_spin; v_seed = fract(i_pos.z * 0.37 + i_pos.x * 0.11);
  v_k = (on ? 1.0 : 0.0) * i_alpha * fogk(i_pos) * (0.35 + 0.65 * i_work);
  vec3 toEye = normalize(u_eye - i_pos);
  vec3 w = i_pos + toEye * r + (u_right * in_corner.x + u_up * in_corner.y) * r * 1.85;
  gl_Position = on ? u_vp * vec4(w, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
}"""
FLAME_FS = """#version 330
in vec2 v_uv; in float v_k; in float v_spin; in float v_seed; out vec4 o;
uniform vec3 u_core, u_mid, u_outer; uniform float u_time; uniform float u_height; uniform float u_int;
// analytic polar flame: tongue height = sum of three phase-shifted |sin| lobes around the ring
void main(){
  float r = length(v_uv); if (r < 0.92 || r > 1.0 + 0.8 * u_height) discard;
  float a = atan(v_uv.y, v_uv.x) + v_spin * 0.4 + v_seed * 6.2832;
  float t = u_time * 3.0;
  float tongue = 0.45 * abs(sin(a * 6.0 + t)) + 0.35 * abs(sin(a * 11.0 - t * 1.3 + 1.7))
               + 0.20 * abs(sin(a * 17.0 + t * 2.1 + 4.1));
  float u = (r - 1.0) / (0.75 * u_height);
  float body = clamp(1.0 - u / (0.25 + 0.9 * tongue), 0.0, 1.0) * smoothstep(-0.12, 0.05, u);
  body *= body;
  float d = (r - 1.0) / 0.05; float ring = exp(-d * d);
  vec3 c = mix(u_core, u_mid, smoothstep(0.0, 0.35, u)); c = mix(c, u_outer, smoothstep(0.35, 0.9, u));
  o = vec4((c * body * 2.6 + u_mid * ring * 1.4) * v_k * u_int, 1.0);
}"""

QUAD_VS = """#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }"""
BRIGHT_FS = """#version 330
in vec2 v_uv; out vec4 o; uniform sampler2D u_tex; uniform vec2 u_px;
vec3 b(vec2 uv){ vec3 c = texture(u_tex, uv).rgb; float l = max(c.r, max(c.g, c.b)); return c * smoothstep(0.55, 1.6, l); }
// downsample to 1/4 res in one go: 4 bilinear taps = a 4x4 box of source texels
void main(){ o = vec4(0.25 * (b(v_uv + u_px * vec2(-1, -1)) + b(v_uv + u_px * vec2(1, -1))
                            + b(v_uv + u_px * vec2(-1, 1)) + b(v_uv + u_px * vec2(1, 1))), 1.0); }"""
BLUR_FS = """#version 330
in vec2 v_uv; out vec4 o; uniform sampler2D u_tex; uniform vec2 u_dir;
// 9-tap gaussian folded into 5 bilinear fetches (weights/offsets merged pairwise)
void main(){ vec3 c = texture(u_tex, v_uv).rgb * 0.227027;
  c += (texture(u_tex, v_uv + u_dir * 1.3846154).rgb + texture(u_tex, v_uv - u_dir * 1.3846154).rgb) * 0.3162162;
  c += (texture(u_tex, v_uv + u_dir * 3.2307692).rgb + texture(u_tex, v_uv - u_dir * 3.2307692).rgb) * 0.0702703;
  o = vec4(c, 1.0); }"""
COMP_FS = """#version 330
in vec2 v_uv; out vec4 o;
uniform sampler2D u_scene, u_b1, u_ui; uniform float u_bloom, u_exposure;
void main(){
  vec3 c = texture(u_scene, v_uv).rgb + texture(u_b1, v_uv).rgb * 1.5 * u_bloom;
  c = vec3(1.0) - exp(-c * u_exposure * 1.6);
  vec2 d = v_uv - 0.5; c *= 1.0 - 0.55 * dot(d, d) * 2.0;
  c = pow(c, vec3(1.0 / 1.05));
  vec4 ui = texture(u_ui, vec2(v_uv.x, 1.0 - v_uv.y));
  o = vec4(mix(c, ui.rgb, ui.a), 1.0);
}"""


def edge_mesh():
    """12 cube edges as camera-facing quads: per vertex (a.xyz, b.xyz, along, side)."""
    corners = [np.array([x, y, z], float) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    out = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.abs(corners[i] - corners[j]).sum() != 2:
                continue
            a, b = corners[i], corners[j]
            for t, s in ((0, -1), (1, -1), (1, 1), (0, -1), (1, 1), (0, 1)):
                out.append([*a, *b, t, s])
    return np.array(out, np.float32)


def ribbon_mesh(n=32):
    v = []
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        v += [(t0, -1), (t1, -1), (t1, 1), (t0, -1), (t1, 1), (t0, 1)]
    return np.array(v, np.float32)


QUAD = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], np.float32)


def rgb(c):
    return tuple(v / 255 for v in c)


class GLRenderer:
    """All GPU state. render() issues exactly 10 draw calls: bg, grid, spheres, edges, ribbons,
    flames, bright/downsample, blur H, blur V, composite."""

    def __init__(self, ctx, size, scale=0.75):
        import moderngl
        self.mgl, self.ctx, self.scale = moderngl, ctx, scale
        P = lambda vs, fs: ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.p_bg, self.p_grid = P(BG_VS, BG_FS), P(GRID_VS, GRID_FS)
        self.p_bake, self.p_skytex = P(BG_VS, BG_BAKE_FS), P(BG_VS, SKYTEX_FS)
        self.p_sph, self.p_edge = P(SPHERE_VS, SPHERE_FS), P(EDGE_VS, EDGE_FS)
        self.p_rib, self.p_flame = P(RIBBON_VS, RIBBON_FS), P(FLAME_VS, FLAME_FS)
        self.p_bright, self.p_blur, self.p_comp = P(QUAD_VS, BRIGHT_FS), P(QUAD_VS, BLUR_FS), P(QUAD_VS, COMP_FS)
        self.quad = ctx.buffer(QUAD)
        self.edges = ctx.buffer(edge_mesh())
        self.rmesh = ctx.buffer(ribbon_mesh(24))
        self.cache = scene3d.InstanceCache()
        self._bufs(self.cache.cap)
        self.n = (0, 0, 0)
        self.draw_calls = 0
        self.ui_key, self.ui_uploads = None, 0
        self.baked_for = None
        self.t_sky = ctx.texture((2048, 1024), 4, dtype="f2"); self.t_sky.repeat_y = False
        self.t_strat = ctx.texture((512, 256), 4, dtype="f2"); self.t_strat.repeat_y = False
        self.f_sky, self.f_strat = ctx.framebuffer([self.t_sky]), ctx.framebuffer([self.t_strat])
        self.resize(size)

    def _bufs(self, cap):
        ctx = self.ctx
        self.cube_buf = ctx.buffer(reserve=4 * scene3d.CUBE_FIELDS * cap, dynamic=True)
        self.flame_buf = ctx.buffer(reserve=4 * scene3d.CUBE_FIELDS * cap, dynamic=True)
        self.rib_buf = ctx.buffer(reserve=4 * scene3d.RIBBON_FIELDS * cap, dynamic=True)
        self.buf_cap = cap
        q = self.quad
        V = lambda p, attrs: ctx.vertex_array(p, attrs, skip_errors=True)
        self.v_bg = V(self.p_bg, [(q, "2f", "in_pos")])
        self.v_bake = V(self.p_bake, [(q, "2f", "in_pos")])
        self.v_skytex = V(self.p_skytex, [(q, "2f", "in_pos")])
        self.v_grid = V(self.p_grid, [(q, "2f", "in_pos")])
        self.v_sph = V(self.p_sph, [(q, "2f", "in_corner"),
            (self.cube_buf, "3f 1f 1f 1f 1f 20x/i", "i_pos", "i_half", "i_state", "i_spin", "i_alpha")])
        self.v_edge = V(self.p_edge, [(self.edges, "3f 3f 1f 1f", "in_a", "in_b", "in_along", "in_side"),
            (self.cube_buf, "3f 1f 1f 4x 1f 3f 1f 1f/i", "i_pos", "i_half", "i_state", "i_alpha", "i_col", "i_style", "i_work")])
        self.v_flame = V(self.p_flame, [(q, "2f", "in_corner"),
            (self.flame_buf, "3f 1f 1f 1f 1f 16x 1f/i", "i_pos", "i_half", "i_state", "i_spin", "i_alpha", "i_work")])
        self.v_rib = V(self.p_rib, [(self.rmesh, "2f", "in_ts"),
            (self.rib_buf, "3f 3f 1f 1f 1f 1f 1f 1f/i", "i_p0", "i_p1", "i_width", "i_act", "i_alpha", "i_state", "i_arc", "i_phase")])
        self.q_bright = V(self.p_bright, [(q, "2f", "in_pos")])
        self.q_blur = V(self.p_blur, [(q, "2f", "in_pos")])
        self.q_comp = V(self.p_comp, [(q, "2f", "in_pos")])

    def resize(self, size):
        ctx = self.ctx
        self.size = w, h = size
        sw, sh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        self.t_scene = ctx.texture((sw, sh), 4, dtype="f2")
        self.t_depth = ctx.depth_renderbuffer((sw, sh))
        self.f_scene = ctx.framebuffer([self.t_scene], self.t_depth)
        qw, qh = max(1, w // 4), max(1, h // 4)   # bloom always at 1/4 of the window
        self.t_b = [ctx.texture((qw, qh), 4, dtype="f2") for _ in range(2)]
        for t in self.t_b:
            t.repeat_x = t.repeat_y = False
        self.f_b = [ctx.framebuffer([t]) for t in self.t_b]
        self.t_ui = ctx.texture((w, h), 4)
        self.ui_key = None

    def bake(self, L):
        """Theme-dependent static textures: star field / cosmic web sky, stratosphere."""
        key = (L.background, L.accent, L.star_density, tuple(L.sky_stops))
        if key == self.baked_for:
            return
        self.baked_for = key
        self.f_sky.use()
        self.set(self.p_bake, u_bg=rgb(L.background), u_web=rgb(L.accent), u_density=max(0.2, L.star_density * 3))
        self.v_bake.render(vertices=6)
        self.t_sky.build_mipmaps()
        self.f_strat.use()
        st = L.sky_stops
        self.set(self.p_skytex, u_s0=rgb(st[0]), u_s1=rgb(st[1]), u_s2=rgb(st[2]), u_s3=rgb(st[3]),
                 u_stars=max(0.3, L.star_density * 2))
        self.v_skytex.render(vertices=6)
        self.t_strat.build_mipmaps()

    def update_instances(self, st, now, L, vp, eye):
        """Rebuild only on change, cull every frame into the preallocated arrays, upload the
        visible slices. Returns True when the instance arrays were rebuilt."""
        c = self.cache
        rebuilt = c.refresh(st, now, L)
        if c.cap > self.buf_cap:
            self._bufs(c.cap)
        n = c.cull(vp, eye)
        if rebuilt or n != self.n or True:
            # culled sets change as the camera moves; uploads are small (<= cap * 48 B)
            if n[0]: self.cube_buf.write(c.vis_cubes[:n[0]])
            if n[1]: self.flame_buf.write(c.vis_flames[:n[1]])
            if n[2]: self.rib_buf.write(c.vis_ribs[:n[2]])
        self.n = n
        return rebuilt

    def set(self, prog, **u):
        for k, v in u.items():
            if k in prog:
                prog[k].value = v

    def set_ui(self, key, draw):
        """Upload the overlay texture only when its content key changed."""
        if key != self.ui_key:
            self.t_ui.write(draw())
            self.ui_key = key
            self.ui_uploads += 1

    def render(self, target, vp, inv, eye, right, up, now, L):
        ctx, mgl = self.ctx, self.mgl
        self.bake(L)
        calls = 0
        nc, nf, nr = self.n
        common = dict(u_vp=tuple(vp.T.flatten()), u_eye=tuple(eye), u_fog=L.fog, u_bg=rgb(L.background), u_time=now)
        self.f_scene.use()
        self.f_scene.depth_mask = True  # clear() honors the mask; it was left False last frame
        self.f_scene.clear(*rgb(L.background), 1.0, depth=1.0)
        ctx.disable(mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
        self.t_sky.use(0)
        self.set(self.p_bg, u_inv=tuple(inv.T.flatten()), u_sky=0)
        self.v_bg.render(vertices=6); calls += 1
        ctx.enable(mgl.BLEND | mgl.DEPTH_TEST)
        ctx.blend_func = mgl.ONE, mgl.ONE
        self.f_scene.depth_mask = False
        self.set(self.p_grid, **common, u_y=-3.6, u_c=(0.0, float(eye[2])), u_col=rgb(L.window_frame))
        self.v_grid.render(vertices=6); calls += 1
        if nc:
            ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
            self.f_scene.depth_mask = True
            self.t_strat.use(1)
            self.set(self.p_sph, **common, u_right=tuple(right), u_up=tuple(up), u_s3=rgb(L.sky_stops[3]), u_strat=1)
            self.v_sph.render(vertices=6, instances=nc); calls += 1
            ctx.blend_func = mgl.ONE, mgl.ONE
            self.f_scene.depth_mask = False
            self.set(self.p_edge, **common, u_glow=0.6 + 0.6 * L.edge_glow, u_lod_far=scene3d.LOD_FAR)
            self.v_edge.render(vertices=72, instances=nc); calls += 1
        if nr:
            cs = L.ribbon_colors if L.ribbon else [L.cx_link] * 3
            cs = [cs[0], cs[len(cs) // 2], cs[-1]]
            self.set(self.p_rib, **common, u_c0=rgb(cs[0]), u_c1=rgb(cs[1]), u_c2=rgb(cs[2]),
                     u_grey=rgb(L.still[0]), u_glow=L.ribbon_glow)
            self.v_rib.render(vertices=24 * 6, instances=nr); calls += 1
        if nf:
            fl = L.flame
            self.set(self.p_flame, **common, u_right=tuple(right), u_up=tuple(up), u_core=rgb(fl[0]),
                     u_mid=rgb(fl[2]), u_outer=rgb(fl[4]), u_height=L.flame_height, u_int=L.flame_intensity)
            self.v_flame.render(vertices=6, instances=nf); calls += 1
        ctx.disable(mgl.DEPTH_TEST | mgl.BLEND)
        # bloom: bright pass straight to 1/4 res, then one H + one V blur
        sw, sh = self.t_scene.size
        qw, qh = self.t_b[0].size
        self.f_b[0].use(); self.t_scene.use(0)
        self.set(self.p_bright, u_tex=0, u_px=(1.0 / sw, 1.0 / sh))
        self.q_bright.render(vertices=6); calls += 1
        self.f_b[1].use(); self.t_b[0].use(0); self.set(self.p_blur, u_tex=0, u_dir=(1.0 / qw, 0.0))
        self.q_blur.render(vertices=6); calls += 1
        self.f_b[0].use(); self.t_b[1].use(0); self.set(self.p_blur, u_dir=(0.0, 1.0 / qh))
        self.q_blur.render(vertices=6); calls += 1
        target.use()
        ctx.viewport = (0, 0, *self.size)
        self.t_scene.use(0); self.t_b[0].use(1); self.t_ui.use(3)
        self.set(self.p_comp, u_scene=0, u_b1=1, u_ui=3, u_bloom=L.bloom, u_exposure=L.exposure)
        self.q_comp.render(vertices=6); calls += 1
        self.draw_calls = calls
        return calls


def camera_mats(cam, size, L):
    w, h = size
    eye, center = cam.eye()
    shift = (1 - viewer.scene_w() / w) if L.panel else 0.0
    proj = scene3d.perspective(48, w / h, 0.1, 600, shift_x=-shift)
    view = scene3d.look_at(eye, center)
    vp = proj @ view
    rot = np.eye(4, dtype=np.float32); rot[:3, :3] = view[:3, :3]
    inv = np.linalg.inv(proj @ rot)
    return vp, inv.astype(np.float32), eye, view[0, :3], view[1, :3]


HINT = "drag orbit - scroll zoom - WASD pan - R reset"


def ui_key(st, now, L, extra=()):
    """Everything the overlay shows; the texture is re-uploaded only when this changes."""
    return (L.label, int(round(st.shown)), st.step, st.status, st.equiv, st.circuit, st.depth, st.cx,
            st.before, st.depth0, st.cx0, st.done, now - st.flash < 1.2, st.demo, *extra)


def draw_ui(pg, surf, st, fs, now, hint_a=0.0, stats=None):
    surf.fill((0, 0, 0, 0))
    viewer.draw_panel(surf, st, fs, now, pg)
    viewer.draw_overlay(surf, st, fs, now, pg)
    small = fs[3]
    if hint_a > 0.01:
        t = small.render(HINT, True, (200, 225, 230)); t.set_alpha(int(255 * hint_a))
        surf.blit(t, (viewer.scene_w() / 2 - t.get_width() / 2, 14))
    if stats:
        t = small.render(stats, True, (150, 200, 205))
        surf.blit(t, (40, 88))
    viewer.draw_frame(surf, pg)
    return pg.image.tostring(surf, "RGBA", False)


class FrameStats:
    def __init__(self):
        self.times, self.text = [], ""

    def tick(self, dt, calls, n, scale):
        self.times = (self.times + [dt])[-60:]
        fps = len(self.times) / max(1e-6, sum(self.times))
        self.text = f"{fps:4.0f} fps  {calls} draw calls  {n[0]} cubes {n[1]} flames {n[2]} ribbons  scale {scale:.2f}"


def frame(rend, target, st, cam, L, now, dt, pg, ui, fs, hint_a=0.0, stats=None):
    """One full frame (also used by the profiler)."""
    vp, inv, eye, right, up = camera_mats(cam, rend.size, L)
    rend.update_instances(st, now, L, vp, eye)
    stx = stats.text if stats else None
    rend.set_ui(ui_key(st, now, L, (round(hint_a, 1), stx)),
                lambda: draw_ui(pg, ui, st, fs, now, hint_a, stx))
    return rend.render(target, vp, inv, eye, right, up, now, L)


def auto_scale(win, fb):
    """Retina-aware internal scale: if the drawable is larger than the window (HiDPI), render the
    3D scene at 0.5 of the framebuffer, else 0.75."""
    return 0.5 if fb[0] > win[0] * 1.5 else 0.75



def _raise_window():
    """macOS often opens a terminal-launched SDL window behind the editor; bring it to the front."""
    try:
        from pygame._sdl2.video import Window
        w = Window.from_display_module()
        w.focus()
        print(f"Legionary viewer window open ({w.size[0]}x{w.size[1]}) - check the Dock if it's hidden", flush=True)
    except Exception as e:  # never block the viewer over window focus
        print(f"Legionary viewer window open (could not raise it: {e})", flush=True)

def run(a, director):
    """Main loop for --renderer 3d (window) or a headless screenshot."""
    import moderngl
    import pygame as pg
    W, H = map(int, a.size.split("x"))
    viewer.W, viewer.H = W, H
    L = viewer.set_look(director.theme, director.label)  # this module's viewer (not __main__)
    L.panel = True  # the 3D scene always uses the slim right panel of big-number cards
    headless = bool(a.screenshot)
    if headless:
        pg.font.init()
        ctx = moderngl.create_standalone_context(require=330)
        target = ctx.framebuffer([ctx.texture((W, H), 4)])
        fb = (W, H)
    else:
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")  # open on-screen, centered
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
        try:
            pg.display.set_mode((W, H), pg.OPENGL | pg.DOUBLEBUF | pg.RESIZABLE, vsync=1)
        except pg.error:
            pg.display.set_mode((W, H), pg.OPENGL | pg.DOUBLEBUF | pg.RESIZABLE)
        pg.display.set_caption("Legionary - live circuit")
        _raise_window()
        ctx = moderngl.create_context()
        target = ctx.screen
        fb = tuple(ctx.screen.viewport[2:]) or (W, H)
    scale = a.render_scale or auto_scale((W, H), fb)
    # the UI/overlay is laid out in window points; the GL target may be the Retina framebuffer
    rend = GLRenderer(ctx, (W, H) if headless else fb, scale)
    q, paused = viewer.make_source(a)
    st, rig, fs = viewer.State(), scene3d.CameraRig(a.angle), viewer.fonts(pg)
    cam = scene3d.CameraController(rig)
    ui = pg.Surface(rend.size, pg.SRCALPHA)
    if rend.size != (W, H):
        viewer.W, viewer.H = rend.size  # draw the overlay at framebuffer resolution
    stats = FrameStats() if a.stats else None
    t0 = last = time.time()
    step = 1 / 30
    dragging, keys = False, set()
    PAN = {pg.K_a: (-1, 0), pg.K_d: (1, 0), pg.K_w: (0, 1), pg.K_s: (0, -1),
           pg.K_LEFT: (-1, 0), pg.K_RIGHT: (1, 0), pg.K_UP: (0, 1), pg.K_DOWN: (0, -1)}
    while True:
        if not headless:
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key in (pg.K_ESCAPE, pg.K_q)):
                    pg.quit(); return 0
                if e.type == pg.KEYDOWN and e.key == pg.K_SPACE:
                    paused.clear() if paused.is_set() else paused.set()
                elif e.type == pg.KEYDOWN and e.key == pg.K_r:
                    cam.reset()
                elif e.type == pg.KEYDOWN and e.key in PAN:
                    keys.add(e.key)
                elif e.type == pg.KEYUP:
                    keys.discard(e.key)
                elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                    dragging = True
                elif e.type == pg.MOUSEBUTTONUP and e.button == 1:
                    dragging = False
                elif e.type == pg.MOUSEMOTION and dragging:
                    cam.drag(*e.rel)
                elif e.type == pg.MOUSEWHEEL:
                    cam.zoom(e.precise_y if hasattr(e, "precise_y") else e.y)
                elif e.type == pg.VIDEORESIZE:
                    W, H = e.w, e.h
                    fb = tuple(ctx.screen.viewport[2:]) or (W, H)
                    fb = (fb[0], fb[1]) if fb[0] >= W else (W, H)
                    rend.resize(fb); ui = pg.Surface(fb, pg.SRCALPHA)
                    viewer.W, viewer.H = fb
                    target = ctx.screen
        now = time.time() - t0
        dt = min(0.1, time.time() - last) or step
        last = time.time()
        if keys:
            px = sum(PAN[k][0] for k in keys); pz = sum(PAN[k][1] for k in keys)
            cam.pan(px, pz, dt)
        if director.poll():
            viewer.set_look(director.theme, director.label)
            L = viewer.look(); L.panel = True
            rend.ui_key = None
        while not q.empty():
            st.apply(q.get(), now)
        st.tick(dt, now)
        rend.cache.refresh(st, now, L)
        rig.update(dt, rend.cache.cubes, st, L.camera_speed)
        cam.update(dt)
        if headless and now < a.screenshot_at:
            time.sleep(step)
            continue
        hint_a = 0.0 if headless else max(0.0, min(1.0, 6.0 - now))  # shows ~5 s, fades over 1 s
        if stats:
            stats.tick(dt, rend.draw_calls or 10, rend.n, rend.scale)
        calls = frame(rend, target, st, cam, L, now, dt, pg, ui, fs, hint_a, stats)
        if headless and stats:  # one-shot frame: redraw so the stats line shows real counts
            stats.tick(dt, calls, rend.n, rend.scale)
            calls = frame(rend, target, st, cam, L, now, dt, pg, ui, fs, hint_a, stats)
        if headless:
            img = pg.image.frombuffer(target.read(components=4), (W, H), "RGBA")
            pg.image.save(pg.transform.flip(img, False, True), a.screenshot)
            print(f"saved {a.screenshot} ({ctx.info['GL_RENDERER']}; {calls} draw calls, scale {rend.scale})")
            return 0
        pg.display.flip()
        time.sleep(max(0, 1 / 60 - (time.time() - last)))  # 60 fps cap (on top of vsync)
