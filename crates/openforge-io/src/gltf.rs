//! glTF 2.0 import/export (GLB container and `.gltf` + external buffers).
//!
//! This is a clean-room implementation targeting the core mesh pipeline:
//! POSITION/NORMAL/TEXCOORD_0 attributes, indexed triangle primitives, and the
//! GLB binary container. It intentionally covers the common subset needed to
//! round-trip geometry rather than the entire spec.

use crate::{finalize_mesh, IoError, IoResult, LoadedScene, MeshData};
use openforge_core::mesh::{Corner, Face, Mesh};
use openforge_core::{Vec2, Vec3};
use std::fs;
use std::path::Path;

const GLB_MAGIC: u32 = 0x46546C67; // "glTF"
const CHUNK_JSON: u32 = 0x4E4F534A; // "JSON"
const CHUNK_BIN: u32 = 0x004E4942; // "BIN\0"

/// Component type constants from the spec.
const F32: u16 = 5126;
const U32: u16 = 5125;
const U16: u16 = 5123;

/// Minimal JSON DOM sufficient for glTF navigation.
enum Dom {
    Null,
    Bool,
    Num(f64),
    Str(String),
    Arr(Vec<Dom>),
    Obj(Vec<(String, Dom)>),
}

impl Dom {
    fn get(&self, key: &str) -> Option<&Dom> {
        match self {
            Dom::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn as_f64(&self) -> Option<f64> {
        match self {
            Dom::Num(n) => Some(*n),
            _ => None,
        }
    }
    fn as_u64(&self) -> Option<u64> {
        self.as_f64().map(|n| n as u64)
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            Dom::Str(s) => Some(s),
            _ => None,
        }
    }
    fn as_arr(&self) -> Option<&[Dom]> {
        match self {
            Dom::Arr(a) => Some(a),
            _ => None,
        }
    }
}

pub fn import(path: &Path) -> IoResult<LoadedScene> {
    let bytes = fs::read(path)?;
    if bytes.len() >= 12 && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == GLB_MAGIC {
        import_glb(&bytes)
    } else {
        let text = String::from_utf8(bytes)
            .map_err(|_| IoError::Parse { format: "gltf", message: "non-utf8 glTF".into() })?;
        import_gltf_str(&text, path.parent())
    }
}

fn import_glb(bytes: &[u8]) -> IoResult<LoadedScene> {
    if bytes.len() < 12 {
        return Err(IoError::Parse { format: "glb", message: "too short".into() });
    }
    let _version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let _total = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let mut cursor = 12usize;
    let mut json_text: Option<String> = None;
    let mut bin: Option<Vec<u8>> = None;
    while cursor + 8 <= bytes.len() {
        let clen = u32::from_le_bytes([bytes[cursor], bytes[cursor + 1], bytes[cursor + 2], bytes[cursor + 3]]) as usize;
        let ctype = u32::from_le_bytes([bytes[cursor + 4], bytes[cursor + 5], bytes[cursor + 6], bytes[cursor + 7]]);
        let start = cursor + 8;
        let end = (start + clen).min(bytes.len());
        let data = &bytes[start..end];
        if ctype == CHUNK_JSON {
            json_text = Some(String::from_utf8_lossy(data).to_string());
        } else if ctype == CHUNK_BIN {
            bin = Some(data.to_vec());
        }
        cursor = start + clen;
        // Chunks are 4-byte aligned; clen already accounts for padding in writers.
    }
    let json = json_text.ok_or(IoError::Parse { format: "glb", message: "no JSON chunk".into() })?;
    build_from_gltf(&json, bin.as_deref(), None)
}

fn import_gltf_str(text: &str, base: Option<&Path>) -> IoResult<LoadedScene> {
    // External .bin buffers are loaded from disk relative to the .gltf file.
    build_from_gltf(text, None, base)
}

fn build_from_gltf(json_text: &str, embedded_bin: Option<&[u8]>, base: Option<&Path>) -> IoResult<LoadedScene> {
    let dom = parse_json(json_text);
    let buffers_dom = dom.get("bufferViews").and_then(|d| d.as_arr()).unwrap_or(&[]);
    let buffers = dom.get("buffers").and_then(|d| d.as_arr()).unwrap_or(&[]);
    let accessors = dom.get("accessors").and_then(|d| d.as_arr()).unwrap_or(&[]);

    // Materialize each buffer's bytes.
    let mut buffer_bytes: Vec<Option<&[u8]>> = Vec::new();
    let mut owned: Vec<Vec<u8>> = Vec::new();
    for (i, buf) in buffers.iter().enumerate() {
        if let Some(uri) = buf.get("uri").and_then(|d| d.as_str()) {
            if let Some(rest) = uri.strip_prefix("data:") {
                let b64 = rest.split(',').nth(1).unwrap_or("");
                let decoded = b64_decode(b64)?;
                owned.push(decoded);
            } else if let Some(base) = base {
                let data = fs::read(base.join(uri))?;
                owned.push(data);
            } else {
                return Err(IoError::Missing("external buffer uri without base path"));
            }
            // Push placeholder; we fix indices after the loop.
            buffer_bytes.push(None);
            let _ = i;
        } else {
            buffer_bytes.push(embedded_bin.map(|b| b as &[u8]));
        }
    }
    // Second pass: assign owned buffers by matching order of uri buffers.
    let mut owned_iter = owned.iter();
    for (i, buf) in buffers.iter().enumerate() {
        if buf.get("uri").is_some() {
            if let Some(data) = owned_iter.next() {
                buffer_bytes[i] = Some(data.as_slice());
            }
        }
    }

    let read_accessor = |index: usize| -> IoResult<Vec<f32>> {
        let acc = accessors.get(index).ok_or(IoError::Missing("accessor"))?;
        let acc_type = acc.get("type").and_then(|d| d.as_str()).unwrap_or("SCALAR");
        let comps = match acc_type {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            _ => 3,
        };
        let count = acc.get("count").and_then(|d| d.as_u64()).unwrap_or(0) as usize;
        let ctype = acc.get("componentType").and_then(|d| d.as_u64()).unwrap_or(F32 as u64) as u16;
        let bv_index = acc.get("bufferView").and_then(|d| d.as_u64()).unwrap_or(0) as usize;
        let bv = buffers_dom.get(bv_index).ok_or(IoError::Missing("bufferView"))?;
        let buf_index = bv.get("buffer").and_then(|d| d.as_u64()).unwrap_or(0) as usize;
        let buf = buffer_bytes.get(buf_index).and_then(|b| *b).ok_or(IoError::Missing("buffer data"))?;
        let bv_offset = bv.get("byteOffset").and_then(|d| d.as_u64()).unwrap_or(0) as usize;
        let acc_offset = acc.get("byteOffset").and_then(|d| d.as_u64()).unwrap_or(0) as usize;
        let start = bv_offset + acc_offset;
        let stride = bv.get("byteStride").and_then(|d| d.as_u64()).map(|s| s as usize);
        let mut out = Vec::with_capacity(count * comps);
        for e in 0..count {
            let base = match stride {
                Some(s) => start + e * s,
                None => start + e * comps * size_of_component(ctype),
            };
            for c in 0..comps {
                let o = base + c * size_of_component(ctype);
                out.push(read_component(buf, o, ctype)?);
            }
        }
        Ok(out)
    };

    let meshes_dom = dom.get("meshes").and_then(|d| d.as_arr()).unwrap_or(&[]);
    let mut meshes: Vec<MeshData> = Vec::new();
    let mut materials: Vec<String> = Vec::new();
    let mats = dom.get("materials").and_then(|d| d.as_arr()).unwrap_or(&[]);
    for m in mats {
        materials.push(m.get("name").and_then(|d| d.as_str()).unwrap_or("material").to_string());
    }

    for (mi, mesh_dom) in meshes_dom.iter().enumerate() {
        let mesh_name = mesh_dom
            .get("name")
            .and_then(|d| d.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("mesh_{}", mi));
        let primitives = mesh_dom.get("primitives").and_then(|d| d.as_arr()).unwrap_or(&[]);
        let mut combined = Mesh::new();
        let mut used_material: Option<String> = None;
        for prim in primitives {
            let attrs = prim.get("attributes").ok_or(IoError::Missing("attributes"))?;
            let pos_acc = attrs.get("POSITION").and_then(|d| d.as_u64()).unwrap_or(0) as usize;
            let positions = read_accessor(pos_acc)?;
            let normals = attrs.get("NORMAL").and_then(|d| d.as_u64()).map(|n| read_accessor(n as usize)).transpose()?;
            let uvs = attrs.get("TEXCOORD_0").and_then(|d| d.as_u64()).map(|n| read_accessor(n as usize)).transpose()?;

            let base_v = combined.positions.len() as u32;
            let mut i = 0;
            while i + 2 < positions.len() {
                combined.add_vertex(Vec3::new(positions[i], positions[i + 1], positions[i + 2]));
                i += 3;
            }
            if let Some(n) = &normals {
                let mut j = 0;
                while j + 2 < n.len() {
                    combined.normals.push(Vec3::new(n[j], n[j + 1], n[j + 2]));
                    j += 3;
                }
            }
            if let Some(t) = &uvs {
                let mut j = 0;
                while j + 1 < t.len() {
                    combined.uvs.push(Vec2::new(t[j], t[j + 1]));
                    j += 2;
                }
            }
            combined.ensure_attributes();

            let ncount = combined.normals.len() as u32;
            let ucount = combined.uvs.len() as u32;

            if let Some(idx_acc) = prim.get("indices").and_then(|d| d.as_u64()) {
                let idx_f = read_accessor(idx_acc as usize)?;
                let idx: Vec<u32> = idx_f.iter().map(|f| *f as u32).collect();
                for tri in idx.chunks(3) {
                    if tri.len() == 3 {
                        let corners = tri
                            .iter()
                            .map(|&local| Corner {
                                position: base_v + local,
                                normal: if ncount > 0 { (base_v + local).min(ncount - 1) } else { 0 },
                                uv: if ucount > 0 { (base_v + local).min(ucount - 1) } else { 0 },
                            })
                            .collect();
                        combined.faces.push(Face { corners, material: 0 });
                    }
                }
            } else {
                // Non-indexed: consecutive triples.
                let nverts = (positions.len() / 3) as u32;
                let mut local = 0u32;
                while local + 2 < nverts {
                    let corners = (0..3)
                        .map(|k| {
                            let p = base_v + local + k;
                            Corner {
                                position: p,
                                normal: if ncount > 0 { p.min(ncount - 1) } else { 0 },
                                uv: if ucount > 0 { p.min(ucount - 1) } else { 0 },
                            }
                        })
                        .collect();
                    combined.faces.push(Face { corners, material: 0 });
                    local += 3;
                }
            }
            if let Some(mat_index) = prim.get("material").and_then(|d| d.as_u64()) {
                used_material = materials.get(mat_index as usize).cloned();
            }
        }
        meshes.push(MeshData { name: mesh_name, mesh: finalize_mesh(combined), material: used_material });
    }

    if meshes.is_empty() {
        return Err(IoError::Parse { format: "gltf", message: "no mesh primitives found".into() });
    }
    Ok(LoadedScene { meshes, materials })
}

fn size_of_component(ctype: u16) -> usize {
    match ctype {
        U16 => 2,
        U32 => 4,
        _ => 4,
    }
}

fn read_component(buf: &[u8], offset: usize, ctype: u16) -> IoResult<f32> {
    match ctype {
        F32 => {
            if offset + 4 > buf.len() {
                return Err(IoError::Parse { format: "gltf", message: "f32 out of range".into() });
            }
            Ok(f32::from_le_bytes([buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]]))
        }
        U32 => {
            if offset + 4 > buf.len() {
                return Err(IoError::Parse { format: "gltf", message: "u32 out of range".into() });
            }
            Ok(u32::from_le_bytes([buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]]) as f32)
        }
        U16 => {
            if offset + 2 > buf.len() {
                return Err(IoError::Parse { format: "gltf", message: "u16 out of range".into() });
            }
            Ok(u16::from_le_bytes([buf[offset], buf[offset + 1]]) as f32)
        }
        _ => Err(IoError::Unsupported("component type")),
    }
}

// --- Export -----------------------------------------------------------------------------

pub fn export(path: &Path, data: &MeshData) -> IoResult<()> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("gltf") {
        // External buffer: reference a .bin file next to the .gltf.
        let bin_name = format!("{}.bin", data.name);
        let (json, bin) = build_gltf_pair(data, Some(&bin_name))?;
        fs::write(path.with_file_name(&bin_name), &bin)?;
        fs::write(path, json)?;
        Ok(())
    } else {
        let glb = build_glb(data)?;
        fs::write(path, glb)?;
        Ok(())
    }
}

fn build_glb(data: &MeshData) -> IoResult<Vec<u8>> {
    // GLB: buffer 0 has no uri; it maps to the BIN chunk.
    let (json, bin) = build_gltf_pair(data, None)?;
    let mut json_bytes = json.into_bytes();
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    let mut bin_bytes = bin;
    while bin_bytes.len() % 4 != 0 {
        bin_bytes.push(0);
    }
    let total = 12 + 8 + json_bytes.len() + 8 + bin_bytes.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(bin_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
    out.extend_from_slice(&bin_bytes);
    Ok(out)
}

/// Build the glTF JSON and binary buffer. `buffer_uri` is `Some` for an
/// external `.bin` reference and `None` for GLB (where the buffer is the BIN
/// chunk and carries no uri).
fn build_gltf_pair(data: &MeshData, buffer_uri: Option<&str>) -> IoResult<(String, Vec<u8>)> {
    let m = &data.mesh;
    let has_uv = !m.uvs.is_empty();
    let has_n = !m.normals.is_empty();

    let mut positions: Vec<u8> = Vec::new();
    for p in &m.positions {
        positions.extend_from_slice(&p.x.to_le_bytes());
        positions.extend_from_slice(&p.y.to_le_bytes());
        positions.extend_from_slice(&p.z.to_le_bytes());
    }
    let mut normal_bytes: Vec<u8> = Vec::new();
    if has_n {
        for n in &m.normals {
            normal_bytes.extend_from_slice(&n.x.to_le_bytes());
            normal_bytes.extend_from_slice(&n.y.to_le_bytes());
            normal_bytes.extend_from_slice(&n.z.to_le_bytes());
        }
    }
    let mut uv_bytes: Vec<u8> = Vec::new();
    if has_uv {
        for t in &m.uvs {
            uv_bytes.extend_from_slice(&t.x.to_le_bytes());
            uv_bytes.extend_from_slice(&t.y.to_le_bytes());
        }
    }
    let mut index_bytes: Vec<u8> = Vec::new();
    for f in &m.faces {
        for c in &f.corners {
            index_bytes.extend_from_slice(&c.position.to_le_bytes());
        }
    }
    let index_count: usize = m.faces.iter().map(|f| f.corners.len()).sum();

    // Concatenate buffer: [positions][normals][uvs][indices].
    let mut bin = Vec::new();
    let pos_offset = 0;
    bin.extend_from_slice(&positions);
    let normal_offset = bin.len();
    bin.extend_from_slice(&normal_bytes);
    let uv_offset = bin.len();
    bin.extend_from_slice(&uv_bytes);
    let index_offset = bin.len();
    bin.extend_from_slice(&index_bytes);

    let (min, max) = bounds_of(m);
    let mut attrs = format!("\"POSITION\":0");
    let mut buffer_views_json = String::new();
    let mut accessors_json = String::new();
    let mut bv_index = 0usize;

    // POSITION bufferView + accessor.
    buffer_views_json.push_str(&format!(
        "{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{}}}",
        positions.len().min(pos_offset + positions.len()) - positions.len() + pos_offset,
        positions.len()
    ));
    accessors_json.push_str(&format!(
        "{{\"bufferView\":{},\"componentType\":{},\"count\":{},\"type\":\"VEC3\",\"min\":{},\"max\":{}}}",
        bv_index,
        F32,
        m.positions.len(),
        format!("[{},{},{}]", min.x, min.y, min.z),
        format!("[{},{},{}]", max.x, max.y, max.z)
    ));
    bv_index += 1;

    if has_n {
        buffer_views_json.push_str(&format!(
            ",{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{}}}",
            normal_offset,
            normal_bytes.len()
        ));
        accessors_json.push_str(&format!(
            ",{{\"bufferView\":{},\"componentType\":{},\"count\":{},\"type\":\"VEC3\"}}",
            bv_index,
            F32,
            m.normals.len()
        ));
        attrs.push_str(",\"NORMAL\":1");
        bv_index += 1;
    }
    if has_uv {
        buffer_views_json.push_str(&format!(
            ",{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{}}}",
            uv_offset,
            uv_bytes.len()
        ));
        accessors_json.push_str(&format!(
            ",{{\"bufferView\":{},\"componentType\":{},\"count\":{},\"type\":\"VEC2\"}}",
            bv_index,
            F32,
            m.uvs.len()
        ));
        attrs.push_str(&format!(",\"TEXCOORD_0\":{}", bv_index));
        bv_index += 1;
    }
    // Indices.
    buffer_views_json.push_str(&format!(
        ",{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{}}}",
        index_offset,
        index_bytes.len()
    ));
    let index_accessor = bv_index;
    accessors_json.push_str(&format!(
        ",{{\"bufferView\":{},\"componentType\":{},\"count\":{},\"type\":\"SCALAR\"}}",
        bv_index, U32, index_count
    ));

    let json = format!(
        "{{\"asset\":{{\"version\":\"2.0\",\"generator\":\"OpenForge\"}},\
         \"buffers\":[{}],\
         \"bufferViews\":[{}],\
         \"accessors\":[{}],\
         \"meshes\":[{{\"name\":\"{}\",\"primitives\":[{{\"attributes\":{{{}}},\"indices\":{}}}]}}],\
         \"nodes\":[{{\"mesh\":0,\"name\":\"{}\"}}],\
         \"scenes\":[{{\"nodes\":[0]}}],\
         \"scene\":0}}",
        match buffer_uri {
            Some(uri) => format!("{{\"byteLength\":{},\"uri\":\"{}\"}}", bin.len(), escape(uri)),
            None => format!("{{\"byteLength\":{}}}", bin.len()),
        },
        buffer_views_json,
        accessors_json,
        escape(&data.name),
        attrs,
        index_accessor,
        escape(&data.name)
    );
    Ok((json, bin))
}

fn bounds_of(m: &Mesh) -> (Vec3, Vec3) {
    let b = m.bounds();
    if b.is_empty() {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        (b.min, b.max)
    }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// --- Minimal JSON parser (recursive descent) ---------------------------------------------

fn parse_json(text: &str) -> Dom {
    let bytes: Vec<char> = text.chars().collect();
    let mut pos = 0usize;
    let dom = parse_value(&bytes, &mut pos);
    dom.unwrap_or(Dom::Null)
}

fn skip_ws(b: &[char], p: &mut usize) {
    while *p < b.len() && b[*p].is_whitespace() {
        *p += 1;
    }
}

fn parse_value(b: &[char], p: &mut usize) -> Option<Dom> {
    skip_ws(b, p);
    if *p >= b.len() {
        return None;
    }
    match b[*p] {
        '{' => parse_object(b, p),
        '[' => parse_array(b, p),
        '"' => parse_string(b, p).map(Dom::Str),
        't' => {
            *p += 4;
            Some(Dom::Bool)
        }
        'f' => {
            *p += 5;
            Some(Dom::Bool)
        }
        'n' => {
            *p += 4;
            Some(Dom::Null)
        }
        _ => parse_number(b, p),
    }
}

fn parse_object(b: &[char], p: &mut usize) -> Option<Dom> {
    *p += 1; // '{'
    let mut fields = Vec::new();
    loop {
        skip_ws(b, p);
        if *p < b.len() && b[*p] == '}' {
            *p += 1;
            break;
        }
        let key = parse_string(b, p)?;
        skip_ws(b, p);
        if *p < b.len() && b[*p] == ':' {
            *p += 1;
        }
        let value = parse_value(b, p)?;
        fields.push((key, value));
        skip_ws(b, p);
        if *p < b.len() && b[*p] == ',' {
            *p += 1;
        }
    }
    Some(Dom::Obj(fields))
}

fn parse_array(b: &[char], p: &mut usize) -> Option<Dom> {
    *p += 1; // '['
    let mut items = Vec::new();
    loop {
        skip_ws(b, p);
        if *p < b.len() && b[*p] == ']' {
            *p += 1;
            break;
        }
        let value = parse_value(b, p)?;
        items.push(value);
        skip_ws(b, p);
        if *p < b.len() && b[*p] == ',' {
            *p += 1;
        }
    }
    Some(Dom::Arr(items))
}

fn parse_string(b: &[char], p: &mut usize) -> Option<String> {
    if *p >= b.len() || b[*p] != '"' {
        return None;
    }
    *p += 1;
    let mut s = String::new();
    while *p < b.len() && b[*p] != '"' {
        if b[*p] == '\\' {
            *p += 1;
            if *p < b.len() {
                s.push(b[*p]);
                *p += 1;
            }
        } else {
            s.push(b[*p]);
            *p += 1;
        }
    }
    *p += 1; // closing quote
    Some(s)
}

fn parse_number(b: &[char], p: &mut usize) -> Option<Dom> {
    let start = *p;
    while *p < b.len() && (b[*p].is_ascii_digit() || b[*p] == '-' || b[*p] == '+' || b[*p] == '.' || b[*p] == 'e' || b[*p] == 'E') {
        *p += 1;
    }
    let text: String = b[start..*p].iter().collect();
    text.parse::<f64>().ok().map(Dom::Num)
}

// --- Base64 (for data: URIs) -------------------------------------------------------------

fn b64_decode(input: &str) -> IoResult<Vec<u8>> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut rev = [255u8; 256];
    for (i, &c) in TABLE.iter().enumerate() {
        rev[c as usize] = i as u8;
    }
    let mut out = Vec::new();
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &ch in input.as_bytes() {
        if ch == b'=' || ch.is_ascii_whitespace() {
            continue;
        }
        let v = rev[ch as usize];
        if v == 255 {
            return Err(IoError::Parse { format: "gltf", message: "invalid base64".into() });
        }
        buf = (buf << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_parser_handles_nested() {
        let dom = parse_json(r#"{"a":[1,2,{"b":"c"}],"d":3.5}"#);
        assert!(dom.get("a").is_some());
        assert_eq!(dom.get("d").and_then(|d| d.as_f64()), Some(3.5));
        let arr = dom.get("a").unwrap().as_arr().unwrap();
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn base64_round_trip() {
        let decoded = b64_decode("aGVsbG8=").unwrap();
        assert_eq!(decoded, b"hello");
    }

    #[test]
    fn glb_round_trip_triangle() {
        let mut mesh = Mesh::new();
        mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[0, 1, 2]);
        mesh.compute_smooth_normals();
        let data = MeshData { name: "tri".into(), mesh, material: None };

        let glb = build_glb(&data).unwrap();
        let loaded = import_bytes(&glb).unwrap();
        assert_eq!(loaded.mesh_count(), 1);
        assert_eq!(loaded.meshes[0].mesh.vertex_count(), 3);
        assert_eq!(loaded.meshes[0].mesh.face_count(), 1);
    }
}

// Small shim so the import_bytes test can reuse the GLB path.
#[cfg(test)]
fn import_bytes(bytes: &[u8]) -> IoResult<LoadedScene> {
    if bytes.len() >= 12 && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == GLB_MAGIC {
        import_glb(bytes)
    } else {
        Err(IoError::Unsupported("not a GLB"))
    }
}
