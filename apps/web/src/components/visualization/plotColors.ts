/** Scientific class identity stays fixed across plots; theme changes affect surface contrast. */
export const CLASS_COLORS = ['#3984BD', '#DF633B', '#4D9568'] as const;
export const CLASS_RGB = [[57,132,189],[223,99,59],[77,149,104]] as const;
export function hexRgb(value: string): readonly number[] {
    const match = /^#([\da-f]{6})$/i.exec(value);
    return match ? [0,2,4].map((offset) => Number.parseInt(match[1].slice(offset,offset+2),16)) : [23,25,27];
}
export function fieldColor(value: number, background: readonly number[]): readonly number[] {
    const bounded = Math.max(0,Math.min(1,value));
    const color = bounded < .5 ? CLASS_RGB[0] : CLASS_RGB[1];
    const strength = Math.abs(bounded-.5)*2;
    return color.map((channel,index) => Math.round(background[index]+(channel-background[index])*strength));
}

export function writeFieldColors(values: ArrayLike<number>, pixels: Uint8ClampedArray, background: readonly number[], domain: readonly [number,number], discrete=false) {
    const range = domain[1]-domain[0] || 1;
    for (let i=0;i<values.length;i++) {
        let value = Math.max(0,Math.min(1,(values[i]-domain[0])/range));
        if (discrete) value = value >= .5 ? 1 : 0;
        const color = value < .5 ? CLASS_RGB[0] : CLASS_RGB[1];
        const strength = Math.abs(value-.5)*2;
        const index = i*4;
        for (let channel=0;channel<3;channel++) pixels[index+channel] = Math.round(background[channel]+(color[channel]-background[channel])*strength);
        pixels[index+3] = 255;
    }
}
