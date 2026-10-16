//! COLLADA (`.dae`) import/export.
//!
//! A hand-written XML subset reader — elements, attributes, text, comments,
//! processing instructions, the five standard entities and numeric character
//! references — carries the parts of the schema a mesh needs:
//! `<library_geometries>` with `<source>` float arrays, `<vertices>`,
//! `<triangles>`, `<polylist>` and `<polygons>`, plus the asset's
//! `<up_axis>` (a `Z_UP` document is turned onto Y on import: `(x, y, z)` →
//! `(x, z, -y)`, the same rule the app's up-axis options use). Per-vertex
//! normals are kept when a primitive indexes them the way it indexes
//! positions; otherwise normals are computed like every other import. Node
//! transforms, effects and animation are out of scope.
//!
//! Export writes one Y-up geometry as index triangles, with a normals source
//! when the mesh's normals are per-vertex.

use crate::{finalize_mesh, IoError, IoResult, LoadedScene, MeshData};
use openforge_core::mesh::Mesh;
use openforge_core::Vec3;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

fn parse_err(message: impl Into<String>) -> IoError {
    IoError::Parse { format: "dae", message: message.into() }
}

// ---------------------------------------------------------------------------
// A minimal XML reader.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Element {
    name: String,
    attrs: Vec<(String, String)>,
    children: Vec<Element>,
    text: String,
}

/// The name without any namespace prefix (`foo:bar` → `bar`).
fn local(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

impl Element {
    fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(key, _)| key == name).map(|(_, value)| value.as_str())
    }

    fn child(&self, name: &str) -> Option<&Element> {
        self.children.iter().find(|child| local(&child.name) == name)
    }

    fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Element> {
        self.children.iter().filter(move |child| local(&child.name) == name)
    }

    fn text_trimmed(&self) -> &str {
        self.text.trim()
    }
}

fn starts_with(chars: &[char], start: usize, needle: &str) -> bool {
    needle.chars().enumerate().all(|(i, c)| chars.get(start + i) == Some(&c))
}

fn find_char(chars: &[char], start: usize, needle: char) -> IoResult<usize> {
    chars[start..]
        .iter()
        .position(|&c| c == needle)
        .map(|i| start + i)
        .ok_or_else(|| parse_err(format!("expected `{needle}`")))
}

fn skip_to(chars: &[char], start: usize, needle: &str) -> IoResult<usize> {
    let mut i = start;
    while i < chars.len() {
        if starts_with(chars, i, needle) {
            return Ok(i + needle.chars().count());
        }
        i += 1;
    }
    Err(parse_err(format!("unterminated `{needle}`")))
}

/// The `>` that ends a tag, skipping quoted attribute values.
fn find_tag_end(chars: &[char], start: usize) -> IoResult<usize> {
    let mut quote: Option<char> = None;
    let mut i = start + 1;
    while i < chars.len() {
        let c = chars[i];
        match quote {
            Some(open) => {
                if c == open {
                    quote = None;
                }
            }
            None => {
                if c == '"' || c == '\'' {
                    quote = Some(c);
                } else if c == '>' {
                    return Ok(i);
                } else if c == '<' {
                    return Err(parse_err("`<` inside a tag"));
                }
            }
        }
        i += 1;
    }
    Err(parse_err("unterminated tag"))
}

fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(position) = rest.find('&') {
        out.push_str(&rest[..position]);
        let tail = &rest[position..];
        let Some(semi) = tail.find(';') else {
            out.push_str(tail);
            return out;
        };
        let entity = &tail[1..semi];
        let decoded = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ if entity.starts_with("#x") || entity.starts_with("#X") => {
                u32::from_str_radix(&entity[2..], 16).ok().and_then(char::from_u32)
            }
            _ if entity.starts_with('#') => entity[1..].parse::<u32>().ok().and_then(char::from_u32),
            _ => None,
        };
        match decoded {
            Some(ch) => {
                out.push(ch);
                rest = &tail[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Parse one tag's innards (`name k="v"`) into a name and decoded attributes.
fn parse_open_tag(body: &str) -> IoResult<(String, Vec<(String, String)>)> {
    let mut chars = body.chars().peekable();
    let mut name = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            break;
        }
        name.push(c);
        chars.next();
    }
    if name.is_empty() {
        return Err(parse_err("empty tag name"));
    }
    let mut attrs = Vec::new();
    loop {
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        if chars.peek().is_none() {
            break;
        }
        let mut key = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_whitespace() || c == '=' {
                break;
            }
            key.push(c);
            chars.next();
        }
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        if chars.next() != Some('=') {
            return Err(parse_err(format!("attribute `{key}` has no `=`")));
        }
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        let quote = chars
            .next()
            .filter(|&c| c == '"' || c == '\'')
            .ok_or_else(|| parse_err(format!("attribute `{key}` has no quoted value")))?;
        let mut value = String::new();
        loop {
            match chars.next() {
                Some(c) if c == quote => break,
                Some(c) => value.push(c),
                None => return Err(parse_err(format!("attribute `{key}` is never closed"))),
            }
        }
        attrs.push((key, decode_entities(&value)));
    }
    Ok((name, attrs))
}

/// Parse a document into its root element.
fn parse_xml(text: &str) -> IoResult<Element> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut position = 0usize;
    let mut stack: Vec<Element> = Vec::new();
    let mut root: Option<Element> = None;
    let mut text_buffer = String::new();

    while position < len {
        let current = chars[position];
        if current != '<' {
            text_buffer.push(current);
            position += 1;
            continue;
        }

        // Text before a tag belongs to the innermost open element.
        let pending = decode_entities(text_buffer.trim());
        if !pending.is_empty() {
            match stack.last_mut() {
                Some(top) => {
                    if !top.text.is_empty() {
                        top.text.push(' ');
                    }
                    top.text.push_str(&pending);
                }
                None => return Err(parse_err("text outside the root element")),
            }
        }
        text_buffer.clear();

        if starts_with(&chars, position, "<?") {
            position = skip_to(&chars, position + 2, "?>")?;
        } else if starts_with(&chars, position, "<!--") {
            position = skip_to(&chars, position + 4, "-->")?;
        } else if starts_with(&chars, position, "<!") {
            // DOCTYPE or another declaration: skip to its `>`.
            position = skip_to(&chars, position + 2, ">")?;
        } else if starts_with(&chars, position, "</") {
            let end = find_char(&chars, position + 2, '>')?;
            let name: String = chars[position + 2..end].iter().collect();
            let name = name.trim().to_string();
            let Some(element) = stack.pop() else {
                return Err(parse_err(format!("closing </{name}> with no open element")));
            };
            if element.name != name {
                return Err(parse_err(format!("</{name}> does not close <{}>", element.name)));
            }
            match stack.last_mut() {
                Some(parent) => parent.children.push(element),
                None => {
                    if root.is_some() {
                        return Err(parse_err("more than one root element"));
                    }
                    root = Some(element);
                }
            }
            position = end + 1;
        } else {
            let end = find_tag_end(&chars, position)?;
            let body: String = chars[position + 1..end].iter().collect();
            let trimmed = body.trim_end();
            let self_closing = trimmed.ends_with('/');
            let body = if self_closing { trimmed[..trimmed.len() - 1].trim_end() } else { trimmed };
            let (name, attrs) = parse_open_tag(body)?;
            let element = Element { name, attrs, children: Vec::new(), text: String::new() };
            if self_closing {
                match stack.last_mut() {
                    Some(parent) => parent.children.push(element),
                    None => {
                        if root.is_some() {
                            return Err(parse_err("more than one root element"));
                        }
                        root = Some(element);
                    }
                }
            } else {
                stack.push(element);
            }
            position = end + 1;
        }
    }

    if !text_buffer.trim().is_empty() {
        if let Some(top) = stack.last_mut() {
            top.text.push_str(text_buffer.trim());
        }
    }
    if let Some(top) = stack.last() {
        return Err(parse_err(format!("<{}> is never closed", top.name)));
    }
    root.ok_or_else(|| parse_err("no root element"))
}

// ---------------------------------------------------------------------------
// Import.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Input {
    semantic: String,
    source: String,
    offset: usize,
}

fn inputs(element: &Element) -> Vec<Input> {
    element
        .children_named("input")
        .map(|input| Input {
            semantic: input.attr("semantic").unwrap_or_default().to_string(),
            source: input
                .attr("source")
                .unwrap_or_default()
                .trim_start_matches('#')
                .to_string(),
            offset: input.attr("offset").and_then(|offset| offset.parse().ok()).unwrap_or(0),
        })
        .collect()
}

/// The position source and its offset: `VERTEX` resolved through the mesh's
/// `<vertices>` node (or taken as the positions source directly), else a
/// `POSITION` input.
fn position_input(inputs: &[Input], vertex_sources: &BTreeMap<String, String>) -> Option<(String, usize)> {
    if let Some(input) = inputs.iter().find(|input| input.semantic == "VERTEX") {
        let source = vertex_sources
            .get(&input.source)
            .cloned()
            .unwrap_or_else(|| input.source.clone());
        return Some((source, input.offset));
    }
    inputs
        .iter()
        .find(|input| input.semantic == "POSITION")
        .map(|input| (input.source.clone(), input.offset))
}

fn parse_floats(text: &str) -> IoResult<Vec<f32>> {
    text.split_whitespace()
        .map(|token| token.parse::<f32>().map_err(|_| parse_err(format!("bad float `{token}`"))))
        .collect()
}

/// Parse an index list (a `<p>`'s text) into integers.
fn parse_indices(text: &str) -> IoResult<Vec<usize>> {
    text.split_whitespace()
        .map(|token| token.parse::<usize>().map_err(|_| parse_err(format!("bad index `{token}`"))))
        .collect()
}

fn index_list(primitive: &Element) -> IoResult<Vec<usize>> {
    let p = primitive
        .child("p")
        .ok_or_else(|| parse_err(format!("a <{}> has no <p>", local(&primitive.name))))?;
    parse_indices(p.text_trimmed())
}

/// Turns source indices of one geometry into mesh vertices and triangles.
struct Builder<'a> {
    sources: &'a BTreeMap<String, Vec<f32>>,
    z_up: bool,
    mesh: Mesh,
    /// `(source id, source index)` → mesh vertex.
    vertex_map: BTreeMap<(String, usize), u32>,
    /// One slot per mesh vertex: the authored normal, when the primitives kept
    /// them the way they kept positions.
    vertex_normal: Vec<Option<Vec3>>,
    triangles: usize,
}

impl Builder<'_> {
    fn vertex(&mut self, source: &str, index: usize, normals: Option<&Vec<f32>>) -> IoResult<u32> {
        if let Some(&vertex) = self.vertex_map.get(&(source.to_string(), index)) {
            return Ok(vertex);
        }
        let floats = self
            .sources
            .get(source)
            .ok_or_else(|| parse_err(format!("unknown source `{source}`")))?;
        let base = index * 3;
        let position = floats
            .get(base..base + 3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .ok_or_else(|| parse_err(format!("index {index} is outside `{source}`")))?;
        let turn = |v: Vec3| if self.z_up { Vec3::new(v.x, v.z, -v.y) } else { v };
        let normal = normals
            .and_then(|floats| floats.get(base..base + 3))
            .map(|n| turn(Vec3::new(n[0], n[1], n[2])));
        let vertex = self.mesh.add_vertex(turn(position));
        self.vertex_normal.push(normal);
        self.vertex_map.insert((source.to_string(), index), vertex);
        Ok(vertex)
    }

    /// One polygon of stride-major source indices, fan-triangulated.
    fn face(&mut self, slots: &[usize], source: &str, position_offset: usize, stride: usize, normals: Option<&Vec<f32>>) -> IoResult<()> {
        let mut vertices = Vec::with_capacity(slots.len() / stride);
        for group in slots.chunks(stride) {
            let index = *group
                .get(position_offset)
                .ok_or_else(|| parse_err("index list is shorter than its inputs"))?;
            vertices.push(self.vertex(source, index, normals)?);
        }
        for i in 1..vertices.len().saturating_sub(1) {
            if self.mesh.add_face(&[vertices[0], vertices[i], vertices[i + 1]]).is_some() {
                self.triangles += 1;
            }
        }
        Ok(())
    }
}

pub fn import(path: &Path) -> IoResult<LoadedScene> {
    let text = fs::read_to_string(path)?;
    import_str(&text)
}

pub fn import_str(text: &str) -> IoResult<LoadedScene> {
    let root = parse_xml(text)?;
    if local(&root.name) != "COLLADA" {
        return Err(parse_err(format!("root element is <{}>, not <COLLADA>", root.name)));
    }
    let z_up = root
        .child("asset")
        .and_then(|asset| asset.child("up_axis"))
        .is_some_and(|axis| axis.text_trimmed().eq_ignore_ascii_case("Z_UP"));

    let library = root
        .child("library_geometries")
        .ok_or_else(|| parse_err("no <library_geometries>"))?;

    let mut meshes = Vec::new();
    let mut materials: Vec<String> = Vec::new();

    for geometry in library.children_named("geometry") {
        let Some(mesh_element) = geometry.child("mesh") else {
            continue;
        };

        let mut sources: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        for source in mesh_element.children_named("source") {
            let (Some(id), Some(array)) = (source.attr("id"), source.child("float_array")) else {
                continue;
            };
            sources.insert(id.to_string(), parse_floats(array.text_trimmed())?);
        }

        let mut vertex_sources: BTreeMap<String, String> = BTreeMap::new();
        for vertices in mesh_element.children_named("vertices") {
            let Some(id) = vertices.attr("id") else {
                continue;
            };
            if let Some(position) = inputs(vertices).into_iter().find(|input| input.semantic == "POSITION") {
                vertex_sources.insert(id.to_string(), position.source);
            }
        }

        let mut builder = Builder {
            sources: &sources,
            z_up,
            mesh: Mesh::new(),
            vertex_map: BTreeMap::new(),
            vertex_normal: Vec::new(),
            triangles: 0,
        };
        let mut primitive_material: Option<String> = None;

        for primitive in mesh_element
            .children
            .iter()
            .filter(|child| matches!(local(&child.name), "triangles" | "polylist" | "polygons"))
        {
            let inputs = inputs(primitive);
            let Some((position_source, position_offset)) = position_input(&inputs, &vertex_sources) else {
                return Err(parse_err(format!(
                    "a <{}> has no VERTEX or POSITION input",
                    local(&primitive.name)
                )));
            };
            if !sources.contains_key(&position_source) {
                return Err(parse_err(format!("unknown source `{position_source}`")));
            }
            let stride = inputs.iter().map(|input| input.offset).max().unwrap_or(0) + 1;
            let position_count = sources[&position_source].len() / 3;
            let normals = inputs
                .iter()
                .find(|input| input.semantic == "NORMAL" && input.offset == position_offset)
                .and_then(|input| sources.get(&input.source))
                .filter(|floats| floats.len() == position_count * 3);

            if let Some(material) = primitive.attr("material") {
                if !materials.iter().any(|known| known == material) {
                    materials.push(material.to_string());
                }
                if primitive_material.is_none() {
                    primitive_material = Some(material.to_string());
                }
            }

            // Each face as one stride-major index run.
            let faces: Vec<Vec<usize>> = match local(&primitive.name) {
                "triangles" => {
                    let p = index_list(primitive)?;
                    let per_triangle = stride * 3;
                    if p.len() % per_triangle != 0 {
                        return Err(parse_err("triangles index array does not divide into triangles"));
                    }
                    p.chunks(per_triangle).map(<[usize]>::to_vec).collect()
                }
                "polylist" => {
                    let p = index_list(primitive)?;
                    let vcount = match primitive.child("vcount") {
                        Some(element) => parse_indices(element.text_trimmed())?,
                        None => Vec::new(),
                    };
                    let mut faces = Vec::new();
                    let mut cursor = 0usize;
                    for count in vcount {
                        let end = cursor + count * stride;
                        if end > p.len() {
                            return Err(parse_err("polylist index array is shorter than its vcount"));
                        }
                        if count >= 3 {
                            faces.push(p[cursor..end].to_vec());
                        }
                        cursor = end;
                    }
                    if cursor != p.len() {
                        return Err(parse_err("polylist index array is longer than its vcount"));
                    }
                    faces
                }
                _ => {
                    let mut faces = Vec::new();
                    for p in primitive.children_named("p") {
                        faces.push(parse_indices(p.text_trimmed())?);
                    }
                    if faces.is_empty() {
                        return Err(parse_err("a <polygons> has no <p>"));
                    }
                    faces
                }
            };

            for face in &faces {
                builder.face(face, &position_source, position_offset, stride, normals)?;
            }
        }

        if builder.triangles == 0 {
            continue;
        }

        let Builder { mut mesh, vertex_normal, .. } = builder;
        if !vertex_normal.is_empty() && vertex_normal.iter().all(Option::is_some) {
            mesh.normals = vertex_normal.into_iter().flatten().collect();
            for face in &mut mesh.faces {
                for corner in &mut face.corners {
                    corner.normal = corner.position;
                }
            }
        }
        let name = geometry
            .attr("name")
            .or_else(|| geometry.attr("id"))
            .unwrap_or("mesh")
            .to_string();
        meshes.push(MeshData {
            name,
            mesh: finalize_mesh(mesh),
            material: primitive_material,
        });
    }

    if meshes.is_empty() {
        return Err(parse_err("no mesh geometry found"));
    }

    Ok(LoadedScene { meshes, materials })
}

// ---------------------------------------------------------------------------
// Export.
// ---------------------------------------------------------------------------

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// A name usable as an XML id: `mesh` plus the name's own word characters.
fn xml_id(name: &str) -> String {
    let mut id = String::from("mesh");
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' {
            id.push(c);
        } else if !id.ends_with('_') {
            id.push('_');
        }
    }
    id.trim_end_matches('_').to_string()
}

fn floats_text(values: impl Iterator<Item = f32>) -> String {
    values.map(|value| value.to_string()).collect::<Vec<_>>().join(" ")
}

fn source_block(id: &str, count: usize, values: String) -> String {
    let mut out = String::new();
    out.push_str(&format!("        <source id=\"{id}\">\n"));
    out.push_str(&format!(
        "          <float_array id=\"{id}_array\" count=\"{}\">{values}</float_array>\n",
        count * 3
    ));
    out.push_str("          <technique_common>\n");
    out.push_str(&format!(
        "            <accessor source=\"#{id}_array\" count=\"{count}\" stride=\"3\">\n"
    ));
    out.push_str("              <param name=\"X\" type=\"float\"/>\n");
    out.push_str("              <param name=\"Y\" type=\"float\"/>\n");
    out.push_str("              <param name=\"Z\" type=\"float\"/>\n");
    out.push_str("            </accessor>\n          </technique_common>\n        </source>\n");
    out
}

pub fn export(path: &Path, data: &MeshData) -> IoResult<()> {
    fs::write(path, export_str(data))?;
    Ok(())
}

pub fn export_str(data: &MeshData) -> String {
    let mesh = &data.mesh;
    let id = xml_id(&data.name);
    let triangles = mesh.triangulate();
    let normals = (!mesh.normals.is_empty() && mesh.normals.len() == mesh.positions.len())
        .then_some(&mesh.normals);

    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
    out.push_str("<COLLADA xmlns=\"http://www.collada.org/2005/11/COLLADASchema\" version=\"1.4.1\">\n");
    out.push_str("  <asset>\n    <up_axis>Y_UP</up_axis>\n  </asset>\n");
    out.push_str("  <library_geometries>\n");
    out.push_str(&format!(
        "    <geometry id=\"{id}_geometry\" name=\"{}\">\n      <mesh>\n",
        escape(&data.name)
    ));
    out.push_str(&source_block(
        &format!("{id}_positions"),
        mesh.positions.len(),
        floats_text(mesh.positions.iter().flat_map(|p| [p.x, p.y, p.z])),
    ));
    if let Some(normals) = normals {
        out.push_str(&source_block(
            &format!("{id}_normals"),
            normals.len(),
            floats_text(normals.iter().flat_map(|n| [n.x, n.y, n.z])),
        ));
    }
    out.push_str(&format!("        <vertices id=\"{id}_vertices\">\n"));
    out.push_str(&format!(
        "          <input semantic=\"POSITION\" source=\"#{id}_positions\"/>\n        </vertices>\n"
    ));
    let material = data
        .material
        .as_ref()
        .map(|name| format!(" material=\"{}\"", escape(name)))
        .unwrap_or_default();
    out.push_str(&format!("        <triangles count=\"{}\"{material}>\n", triangles.len()));
    out.push_str(&format!(
        "          <input semantic=\"VERTEX\" source=\"#{id}_vertices\" offset=\"0\"/>\n"
    ));
    if normals.is_some() {
        out.push_str(&format!(
            "          <input semantic=\"NORMAL\" source=\"#{id}_normals\" offset=\"0\"/>\n"
        ));
    }
    out.push_str("          <p>");
    out.push_str(&floats_text(triangles.iter().flat_map(|triangle| triangle.iter().map(|&i| i as f32))));
    out.push_str("</p>\n        </triangles>\n      </mesh>\n    </geometry>\n  </library_geometries>\n");
    out.push_str("  <library_visual_scenes>\n    <visual_scene id=\"OpenForge_Scene\">\n");
    out.push_str(&format!(
        "      <node id=\"{id}_node\" name=\"{}\">\n        <instance_geometry url=\"#{id}_geometry\"/>\n      </node>\n",
        escape(&data.name)
    ));
    out.push_str("    </visual_scene>\n  </library_visual_scenes>\n");
    out.push_str("  <scene>\n    <instance_visual_scene url=\"#OpenForge_Scene\"/>\n  </scene>\n");
    out.push_str("</COLLADA>\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> Mesh {
        let mut mesh = Mesh::new();
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            mesh.add_vertex(Vec3::new(x, y, 0.0));
        }
        mesh.add_face(&[0, 1, 2, 3]);
        mesh.compute_smooth_normals();
        mesh
    }

    /// The refusal a document earns, as a string.
    fn refusal(text: &str) -> String {
        match import_str(text) {
            Ok(loaded) => panic!("expected a refusal, got {} mesh(es)", loaded.mesh_count()),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn export_then_import_returns_the_quad_triangulated() {
        let data = MeshData { name: "Quad Mesh".into(), mesh: quad(), material: Some("Red".into()) };
        let text = export_str(&data);
        assert!(text.contains("<up_axis>Y_UP</up_axis>"));
        assert!(text.contains("name=\"Quad Mesh\""));
        assert!(text.contains("semantic=\"NORMAL\""));
        assert!(text.contains("material=\"Red\""));

        let loaded = import_str(&text).unwrap();
        assert_eq!(loaded.mesh_count(), 1);
        assert_eq!(loaded.meshes[0].name, "Quad Mesh");
        assert_eq!(loaded.meshes[0].material.as_deref(), Some("Red"));
        assert_eq!(loaded.materials, vec!["Red"]);
        let mesh = &loaded.meshes[0].mesh;
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 2, "the quad came back as triangles");
        assert_eq!(mesh.positions[2], Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(mesh.normals.len(), 4, "the authored normals were kept");
    }

    #[test]
    fn a_hand_written_z_up_polylist_turns_onto_y() {
        let text = r##"<?xml version="1.0"?>
<!-- a Z-up quad fanned from a polylist -->
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
  <asset><up_axis>Z_UP</up_axis></asset>
  <library_geometries>
    <geometry id="tri-1" name="Tri&amp;Co">
      <mesh>
        <source id="tri-pos">
          <float_array id="tri-pos-a" count="12">0 0 0  1 0 0  0 1 0  0 0 1</float_array>
        </source>
        <vertices id="tri-v"><input semantic="POSITION" source="#tri-pos"/></vertices>
        <polylist count="1">
          <input semantic="VERTEX" source="#tri-v" offset="0"/>
          <vcount>4</vcount>
          <p>0 1 2 3</p>
        </polylist>
      </mesh>
    </geometry>
  </library_geometries>
</COLLADA>
"##;
        let loaded = import_str(text).unwrap();
        assert_eq!(loaded.meshes[0].name, "Tri&Co", "the entity was decoded");
        let mesh = &loaded.meshes[0].mesh;
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 2, "the quad fanned");
        assert_eq!(mesh.positions[1], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(mesh.positions[3], Vec3::new(0.0, 1.0, 0.0), "Z-up (0,0,1) is Y-up (0,1,0)");
        assert_eq!(mesh.normals.len(), 4, "normals were computed");
    }

    #[test]
    fn stride_polygons_and_materials_are_read() {
        let text = r##"<COLLADA>
  <library_geometries><geometry id="g" name="Two"><mesh>
    <source id="pos"><float_array id="pa" count="12">0 0 0  1 0 0  0 1 0  0 0 1</float_array></source>
    <source id="nrm"><float_array id="na" count="12">0 0 1  0 0 1  0 0 1  0 0 1</float_array></source>
    <vertices id="v"><input semantic="POSITION" source="#pos"/></vertices>
    <triangles count="1" material="Blue">
      <input semantic="VERTEX" source="#v" offset="0"/>
      <input semantic="NORMAL" source="#nrm" offset="1"/>
      <p>0 0  1 1  2 2</p>
    </triangles>
    <polygons material="Blue">
      <input semantic="VERTEX" source="#v" offset="0"/>
      <p>0 1 2 3</p>
    </polygons>
  </mesh></geometry></library_geometries>
</COLLADA>"##;
        let loaded = import_str(text).unwrap();
        assert_eq!(loaded.materials, vec!["Blue"], "one name, listed once");
        assert_eq!(loaded.meshes[0].material.as_deref(), Some("Blue"));
        let mesh = &loaded.meshes[0].mesh;
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 3, "one strided triangle and a fanned quad");
        // The normal input sits at another offset, so it is not the vertex
        // normals; the mesh got computed ones instead.
        assert_eq!(mesh.normals.len(), 4);
    }

    #[test]
    fn broken_documents_are_refused() {
        let cases = [
            ("<COLLADA>", "never closed"),
            ("<html></html>", "not <COLLADA>"),
            ("<COLLADA><asset/></COLLADA>", "no <library_geometries>"),
            (
                "<COLLADA><library_geometries></library_geometries></COLLADA>",
                "no mesh geometry found",
            ),
            (
                "<COLLADA><asset></COLLADA>",
                "does not close",
            ),
            (
                "<COLLADA><library_geometries><geometry><mesh><triangles><p>0 1 2</p></triangles></mesh></geometry></library_geometries></COLLADA>",
                "no VERTEX or POSITION input",
            ),
            (
                "<COLLADA><library_geometries><geometry><mesh>\
                 <source id=\"p\"><float_array>0 0 0</float_array></source>\
                 <vertices id=\"v\"><input semantic=\"POSITION\" source=\"#p\"/></vertices>\
                 <triangles><input semantic=\"VERTEX\" source=\"#v\"/><p>0 1 2</p></triangles>\
                 </mesh></geometry></library_geometries></COLLADA>",
                "index 1 is outside",
            ),
            (
                "<COLLADA><library_geometries><geometry><mesh>\
                 <source id=\"p\"><float_array>0 0 0 1 0 0 0 1 0</float_array></source>\
                 <vertices id=\"v\"><input semantic=\"POSITION\" source=\"#p\"/></vertices>\
                 <triangles><input semantic=\"VERTEX\" source=\"#v\"/><p>0 1</p></triangles>\
                 </mesh></geometry></library_geometries></COLLADA>",
                "does not divide into triangles",
            ),
            (
                "<COLLADA><library_geometries><geometry><mesh>\
                 <source id=\"p\"><float_array>0 0 x</float_array></source>\
                 <vertices id=\"v\"><input semantic=\"POSITION\" source=\"#p\"/></vertices>\
                 </mesh></geometry></library_geometries></COLLADA>",
                "bad float `x`",
            ),
        ];
        for (text, message) in cases {
            let error = refusal(text);
            assert!(error.contains(message), "{error:?} should mention {message:?}");
        }
    }

    #[test]
    fn xml_ids_stay_usable_and_attributes_are_escaped() {
        let data = MeshData {
            name: "pörşan & <rig>".into(),
            mesh: quad(),
            material: Some("A&B".into()),
        };
        let text = export_str(&data);
        assert!(text.contains("id=\"meshp_r_an_rig_geometry\""), "{text}");
        assert!(text.contains("name=\"pörşan &amp; &lt;rig&gt;\""));
        assert!(text.contains("material=\"A&amp;B\""));
        let loaded = import_str(&text).unwrap();
        assert_eq!(loaded.meshes[0].name, "pörşan & <rig>");
        assert_eq!(loaded.meshes[0].material.as_deref(), Some("A&B"));
    }
}
